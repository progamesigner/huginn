from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from datetime import datetime
from typing import cast
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import or_, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.features.conversations.entities import Conversation, Part, Turn
from app.features.conversations.exceptions import NoTurnsToCompactError
from app.features.conversations.sources.sqlalchemy import (
    ConversationRecord,
    PartRecord,
    TurnRecord,
)
from lib.guid import GUIDContract
from lib.temporal import ClockContract


def _to_conversation(record: ConversationRecord) -> Conversation:
    return Conversation(
        id=record.id,
        channel=record.channel,
        chat_id=record.chat_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        compact_version=record.compact_version,
        summarized_through_turn_id=record.summarized_through_turn_id,
        reset_through_turn_id=record.reset_through_turn_id,
    )


def _to_part(record: PartRecord) -> Part:
    return Part(
        role=record.role,
        kind=record.kind,
        payload=cast(Mapping[str, JsonValue], record.payload),
    )


def _to_turn(record: TurnRecord) -> Turn:
    return Turn(
        id=record.id,
        agent=record.agent,
        parts=tuple(_to_part(part) for part in record.parts),
        usage=cast(Mapping[str, JsonValue], record.usage),
        created_at=record.created_at,
    )


class ConversationRepository:
    clock: ClockContract
    guid: GUIDContract
    session: AsyncSession

    def __init__(
        self,
        session: AsyncSession,
        guid: GUIDContract,
        clock: ClockContract,
    ) -> None:
        self.guid = guid
        self.session = session
        self.clock = clock

    async def get_or_create(
        self,
        channel: str,
        chat_id: str,
    ) -> Conversation:
        query = select(ConversationRecord).where(
            ConversationRecord.channel == channel,
            ConversationRecord.chat_id == chat_id,
        )
        record = (await self.session.execute(query)).scalar_one_or_none()

        if record is None:
            now = self.clock.now()
            record = ConversationRecord(
                id=self.guid.generate(),
                channel=channel,
                chat_id=chat_id,
                created_at=now,
                updated_at=now,
            )
            self.session.add(record)
            await self.session.flush()

        return _to_conversation(record)

    async def find_by_chat(
        self,
        channel: str,
        chat_id: str,
    ) -> Conversation | None:
        query = select(ConversationRecord).where(
            ConversationRecord.channel == channel,
            ConversationRecord.chat_id == chat_id,
        )
        record = (await self.session.execute(query)).scalar_one_or_none()

        return None if record is None else _to_conversation(record)

    async def append_turn(
        self,
        conversation: Conversation,
        agent: str,
        parts: Sequence[Part],
        usage: Mapping[str, JsonValue] | None = None,
    ) -> Turn:
        record = TurnRecord(
            id=self.guid.generate(),
            conversation_id=conversation.id,
            agent=agent,
            usage=dict(usage or {}),
            parts=[
                PartRecord(
                    id=self.guid.generate(),
                    seq=index,
                    role=part.role,
                    kind=part.kind,
                    payload=dict(part.payload),
                )
                for index, part in enumerate(parts)
            ],
        )
        self.session.add(record)
        await self.session.flush()

        return _to_turn(record)

    async def latest_agent(self, conversation: Conversation) -> str | None:
        query = (
            select(TurnRecord.agent)
            .where(TurnRecord.conversation_id == conversation.id)
            .order_by(TurnRecord.id.desc())
            .limit(1)
        )
        return (await self.session.execute(query)).scalar_one_or_none()

    async def load_recent_turns(
        self,
        conversation: Conversation,
        limit: int,
    ) -> tuple[Turn, ...]:
        query = (
            select(TurnRecord)
            .where(TurnRecord.conversation_id == conversation.id)
            # NOTE: created_at is a transaction clock, so turns written together tie;
            # the monotonic UUIDv7 id breaks the tie to keep ordering deterministic
            .order_by(TurnRecord.created_at.desc(), TurnRecord.id.desc())
            .limit(limit)
            .options(selectinload(TurnRecord.parts))
        )
        records = (await self.session.execute(query)).scalars().all()

        return tuple(_to_turn(record) for record in reversed(records))

    async def routing_samples(
        self,
        channel: str,
        since: datetime,
        history_turns: int,
    ) -> AsyncIterator[tuple[Turn, tuple[Turn, ...]]]:
        query = (
            select(TurnRecord)
            .join(
                ConversationRecord,
                ConversationRecord.id == TurnRecord.conversation_id,
            )
            .where(
                ConversationRecord.channel == channel,
                TurnRecord.created_at >= since,
                TurnRecord.parts.any(PartRecord.role == 'user'),
            )
            .order_by(TurnRecord.created_at, TurnRecord.id)
            .options(selectinload(TurnRecord.parts))
        )
        # NOTE: keyset pages, so the archive is never loaded whole and no server cursor
        # stays open while a second query rebuilds each sample's prior context
        cursor = None
        while True:
            page = (
                query
                if cursor is None
                else query.where(tuple_(TurnRecord.created_at, TurnRecord.id) > cursor)
            )
            records = (await self.session.execute(page.limit(100))).scalars().all()
            if not records:
                break
            for record in records:
                prior_query = (
                    select(TurnRecord)
                    .where(
                        TurnRecord.conversation_id == record.conversation_id,
                        tuple_(TurnRecord.created_at, TurnRecord.id)
                        < (record.created_at, record.id),
                        TurnRecord.parts.any(PartRecord.role == 'user'),
                    )
                    .order_by(TurnRecord.created_at.desc(), TurnRecord.id.desc())
                    .limit(history_turns)
                    .options(selectinload(TurnRecord.parts))
                )
                prior = (
                    (await self.session.execute(prior_query)).scalars().all()
                    if history_turns
                    else []
                )
                yield (
                    _to_turn(record),
                    tuple(_to_turn(item) for item in reversed(prior)),
                )
            cursor = (records[-1].created_at, records[-1].id)

    async def load_routing_turns(
        self,
        conversation: Conversation,
        limit: int,
    ) -> tuple[Turn, ...]:
        if limit <= 0:
            return ()
        query = (
            select(TurnRecord)
            .where(
                TurnRecord.conversation_id == conversation.id,
                TurnRecord.parts.any(PartRecord.role == 'user'),
            )
            .order_by(TurnRecord.created_at.desc(), TurnRecord.id.desc())
            .limit(limit)
            .options(selectinload(TurnRecord.parts))
        )
        records = (await self.session.execute(query)).scalars().all()
        return tuple(_to_turn(record) for record in reversed(records))

    async def load_turns_since_summary(
        self,
        conversation: Conversation,
    ) -> tuple[Turn, ...]:
        query = (
            select(TurnRecord)
            .where(
                TurnRecord.conversation_id == conversation.id,
                ~TurnRecord.parts.any(PartRecord.kind == 'summary'),
            )
            .order_by(TurnRecord.id.asc())
            .options(selectinload(TurnRecord.parts))
        )
        if (boundary_id := conversation.summarized_through_turn_id) is not None:
            query = query.where(TurnRecord.id > boundary_id)

        records = (await self.session.execute(query)).scalars().all()

        return tuple(_to_turn(record) for record in records)

    async def referenced(self, ids: Collection[UUID]) -> frozenset[UUID]:
        keys = [str(id_) for id_ in ids]
        if not keys:
            return frozenset()

        blob = PartRecord.payload['blob'].as_string()
        extracted = PartRecord.payload['extracted_blob'].as_string()
        rows = await self.session.execute(
            select(blob, extracted).where(or_(blob.in_(keys), extracted.in_(keys))),
        )
        return frozenset(UUID(value) for row in rows for value in row if value in keys)

    async def load_document_blob_ids(
        self,
        conversation: Conversation,
    ) -> frozenset[str]:
        query = (
            select(PartRecord.payload)
            .join(TurnRecord, PartRecord.turn_id == TurnRecord.id)
            .where(TurnRecord.conversation_id == conversation.id)
        )
        if (reset_id := conversation.reset_through_turn_id) is not None:
            query = query.where(TurnRecord.id > reset_id)
        payloads = (await self.session.execute(query)).scalars().all()
        return frozenset(
            value
            for payload in (cast(Mapping[str, JsonValue], raw) for raw in payloads)
            if isinstance(value := payload.get('extracted_blob'), str)
        )

    async def load_current_summary(
        self,
        conversation: Conversation,
    ) -> Mapping[str, JsonValue] | None:
        query = (
            select(PartRecord)
            .join(TurnRecord, PartRecord.turn_id == TurnRecord.id)
            .where(
                TurnRecord.conversation_id == conversation.id,
                PartRecord.kind == 'summary',
            )
            .order_by(PartRecord.id.desc())
            .limit(1)
        )
        # NOTE: compaction writes the summary in a turn after the boundary it set, so a
        # summary at or below the boundary was cut off by a reset and must stay cut
        if (boundary_id := conversation.summarized_through_turn_id) is not None:
            query = query.where(TurnRecord.id > boundary_id)
        part = (await self.session.execute(query)).scalar_one_or_none()

        return cast(Mapping[str, JsonValue], part.payload) if part is not None else None

    async def reset(
        self,
        conversation: Conversation,
        expected_version: int,
    ) -> Conversation | None:
        newest = (
            select(TurnRecord.id)
            .where(TurnRecord.conversation_id == conversation.id)
            .order_by(TurnRecord.id.desc())
            .limit(1)
            .scalar_subquery()
        )
        statement = (
            update(ConversationRecord)
            .where(
                ConversationRecord.id == conversation.id,
                ConversationRecord.compact_version == expected_version,
            )
            .values(
                compact_version=ConversationRecord.compact_version + 1,
                summarized_through_turn_id=newest,
                reset_through_turn_id=newest,
                updated_at=self.clock.now(),
            )
            .returning(ConversationRecord)
            .execution_options(synchronize_session=False)
        )
        record = (await self.session.execute(statement)).scalar_one_or_none()

        return None if record is None else _to_conversation(record)

    async def apply_compaction(
        self,
        conversation: Conversation,
        summary_payload: Mapping[str, JsonValue],
        turns_folded: tuple[Turn, ...],
        expected_version: int,
    ) -> Conversation | None:
        if not turns_folded:
            raise NoTurnsToCompactError(
                'apply_compaction requires at least one folded turn',
            )

        boundary = turns_folded[-1]
        statement = (
            update(ConversationRecord)
            .where(
                ConversationRecord.id == conversation.id,
                ConversationRecord.compact_version == expected_version,
            )
            .values(
                compact_version=ConversationRecord.compact_version + 1,
                summarized_through_turn_id=boundary.id,
                updated_at=self.clock.now(),
            )
            .returning(ConversationRecord)
            .execution_options(synchronize_session=False)
        )
        record = (await self.session.execute(statement)).scalar_one_or_none()
        if record is None:
            return None  # CAS conflict: another compact already won; insert nothing

        self.session.add(
            TurnRecord(
                id=self.guid.generate(),
                conversation_id=conversation.id,
                agent=boundary.agent,
                usage={},
                parts=[
                    PartRecord(
                        id=self.guid.generate(),
                        seq=0,
                        role='assistant',
                        kind='summary',
                        payload=dict(summary_payload),
                    ),
                ],
            ),
        )
        await self.session.flush()

        return _to_conversation(record)
