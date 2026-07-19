from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.features.conversations.types import Conversation, Part, Turn
from lib.guid import GUIDContract

from .sqlalchemy import ConversationRecord, PartRecord, TurnRecord


def _to_conversation(record: ConversationRecord) -> Conversation:
    return Conversation(
        id=record.id,
        channel=record.channel,
        chat_id=record.chat_id,
        agent=record.agent,
        created_at=record.created_at,
        updated_at=record.updated_at,
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
        parts=tuple(_to_part(part) for part in record.parts),
        usage=cast(Mapping[str, JsonValue], record.usage),
        created_at=record.created_at,
    )


class ConversationRepository:
    guid: GUIDContract
    session: AsyncSession

    def __init__(self, session: AsyncSession, guid: GUIDContract) -> None:
        self.guid = guid
        self.session = session

    async def get_or_create(
        self,
        channel: str,
        chat_id: str,
        agent: str,
    ) -> Conversation:
        query = select(ConversationRecord).where(
            ConversationRecord.channel == channel,
            ConversationRecord.chat_id == chat_id,
            ConversationRecord.agent == agent,
        )
        record = (await self.session.execute(query)).scalar_one_or_none()

        if record is None:
            record = ConversationRecord(
                id=self.guid.generate(),
                channel=channel,
                chat_id=chat_id,
                agent=agent,
            )
            self.session.add(record)
            await self.session.flush()

        return _to_conversation(record)

    async def append_turn(
        self,
        conversation: Conversation,
        parts: Sequence[Part],
        usage: Mapping[str, JsonValue] | None = None,
    ) -> Turn:
        record = TurnRecord(
            id=self.guid.generate(),
            conversation_id=conversation.id,
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
