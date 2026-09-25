import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from typing import override

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config.registry import HuginnConfig
from app.features.agents.config import (
    AGENTS,
    DEFAULTS,
    load_instructions,
    resolve_selection,
)
from app.features.agents.contracts import TranscriptStoreContract
from app.features.agents.types import (
    PromptAttachment,
    RoutingContext,
    RoutingHistory,
    RoutingTurn,
    TurnPrompt,
)
from app.features.blobs.config import BLOBS, RetentionConfig
from app.features.blobs.contracts import BlobStoreContract, BlobStoreFactoryContract
from lib.database import DatabaseContract
from lib.guid import GUIDContract
from lib.llm import LLMContract, Message, ModelRef
from lib.temporal import ClockContract

from .compaction import (
    CompactionService,
    SummaryOutput,
    replay_messages,
    summary_message,
)
from .config import COMPACTION
from .contracts import ConversationControlContract
from .repositories import ConversationRepository
from .types import Conversation, Part, Turn


def routing_turn(turn: Turn) -> RoutingTurn:
    user = []
    answer = []
    attachments = []
    for part in turn.parts:
        text = part.payload.get('text')
        if part.role == 'user':
            if part.kind in ('text', 'audio') and isinstance(text, str):
                user.append(text)
            if part.kind in ('audio', 'document', 'image', 'video'):
                attachments.append(part.kind)
        elif part.role == 'assistant' and part.kind == 'text' and isinstance(text, str):
            answer.append(text)
    return RoutingTurn(turn.agent, tuple(user), '\n'.join(answer), tuple(attachments))


_TEXT = 'text'

logger = logging.getLogger(__name__)


class ConversationTranscript:
    blobs: BlobStoreContract
    conversation: Conversation
    repository: ConversationRepository
    retention: RetentionConfig

    def __init__(
        self,
        repository: ConversationRepository,
        blobs: BlobStoreContract,
        conversation: Conversation,
        retention: RetentionConfig,
    ) -> None:
        self.blobs = blobs
        self.conversation = conversation
        self.repository = repository
        self.retention = retention

    async def history(self) -> tuple[Message, ...]:
        payload = await self.repository.load_current_summary(self.conversation)
        turns = await self.repository.load_turns_since_summary(self.conversation)
        summary = (
            None
            if payload is None
            else SummaryOutput.model_validate(payload, strict=False)
        )

        return (
            *(() if summary is None else (summary_message(summary),)),
            *replay_messages(turns),
        )

    async def record(
        self,
        agent: str,
        prompts: Sequence[TurnPrompt],
        answer: str,
    ) -> None:
        parts = []
        for prompt in prompts:
            if prompt.text:
                parts.append(
                    Part(
                        role='user',
                        kind=_TEXT,
                        payload={'text': prompt.text, 'actor': prompt.actor},
                    ),
                )
            for attachment in prompt.attachments:
                parts.append(await self._media(attachment, prompt.actor))

        if answer:
            parts.append(Part(role='assistant', kind=_TEXT, payload={'text': answer}))

        await self.repository.append_turn(self.conversation, agent, parts)

    async def _media(self, attachment: PromptAttachment, actor: str) -> Part:
        blob = await self.blobs.store(
            attachment.mime,
            attachment.data,
            kind=attachment.kind,
            filename=attachment.name,
            ttl=self.retention.inbound,
        )
        extracted_blob = None
        if attachment.document_text is not None:
            extracted = await self.blobs.store(
                'text/markdown',
                attachment.document_text.encode(),
                kind='document',
                filename=attachment.name,
                ttl=self.retention.generated,
            )
            extracted_blob = str(extracted.id)

        return Part(
            role='user',
            kind=attachment.kind,
            payload={
                'text': attachment.transcript or attachment.hint,
                'actor': actor,
                'blob': str(blob.id),
                'sha256': blob.sha256,
                'mime': blob.mime,
                'size': blob.size,
                'reference': attachment.reference,
                'unique': attachment.unique,
                'extracted_blob': extracted_blob,
            },
        )


class ConversationControl(ConversationControlContract):
    clock: ClockContract
    config: HuginnConfig
    database: DatabaseContract
    guid: GUIDContract
    llm: LLMContract

    def __init__(
        self,
        database: DatabaseContract,
        guid: GUIDContract,
        clock: ClockContract,
        llm: LLMContract,
        config: HuginnConfig,
    ) -> None:
        self.clock = clock
        self.config = config
        self.database = database
        self.guid = guid
        self.llm = llm

    @override
    async def compact(self, channel: str, chat: str) -> bool:
        async with self.database.session() as session, session.begin():
            repository = ConversationRepository(session, self.guid, self.clock)
            conversation = await repository.find_by_chat(channel, chat)
            if conversation is None:
                return False
            agent = await repository.latest_agent(conversation)
            configured = (
                None if agent is None else self.config.section(AGENTS).root.get(agent)
            )
            if configured is None:
                logger.warning(
                    'Conversation %s has no configured latest agent %r',
                    conversation.id,
                    agent,
                )
                return False

            ref = resolve_selection(
                configured.provider,
                configured.model,
                self.config.section(DEFAULTS),
            )
            service = CompactionService(
                self.llm,
                repository,
                instructions=load_instructions(configured),
                instruction=self.config.section(COMPACTION).summary_prompt,
            )
            result = await service.force(conversation, ref)

        return result.compacted

    @override
    async def reset(self, channel: str, chat: str) -> bool:
        async with self.database.session() as session, session.begin():
            repository = ConversationRepository(session, self.guid, self.clock)
            conversation = await repository.find_by_chat(channel, chat)
            if conversation is None:
                return False
            reset = await repository.reset(conversation, conversation.compact_version)

        return reset is not None


class ConversationTranscriptStore(TranscriptStoreContract):
    blobs: BlobStoreFactoryContract
    clock: ClockContract
    config: HuginnConfig
    database: DatabaseContract
    guid: GUIDContract
    llm: LLMContract

    def __init__(
        self,
        database: DatabaseContract,
        guid: GUIDContract,
        clock: ClockContract,
        llm: LLMContract,
        config: HuginnConfig,
        blobs: BlobStoreFactoryContract,
    ) -> None:
        self.blobs = blobs
        self.clock = clock
        self.config = config
        self.database = database
        self.guid = guid
        self.llm = llm

    @override
    async def history(
        self,
        channel: str,
        chat: str,
    ) -> tuple[Message, ...]:
        async with self._open(channel, chat) as transcript:
            return await transcript.history()

    @override
    async def routing_samples(
        self,
        channel: str,
        since: datetime,
        history_turns: int,
    ) -> AsyncIterator[RoutingHistory]:
        async with self.database.session() as session:
            repository = self._repository(session)
            async for turn, preceding in repository.routing_samples(
                channel,
                since,
                history_turns,
            ):
                current = routing_turn(turn)
                prompts = tuple(
                    TurnPrompt(text=text, actor='') for text in current.user
                )
                if current.attachments:
                    prompts += (
                        TurnPrompt(
                            text='[attachments: '
                            + '; '.join(current.attachments)
                            + ']',
                            actor='',
                        ),
                    )
                # NOTE: a summary is rewritten in place, so today's says nothing about
                # what the router saw back then
                yield RoutingHistory(
                    str(turn.id),
                    turn.agent,
                    RoutingContext(
                        prompts,
                        tuple(routing_turn(prior) for prior in preceding),
                    ),
                    turn.created_at,
                    preceding[-1].created_at if preceding else None,
                )

    @override
    async def recent_turns(
        self,
        channel: str,
        chat: str,
        limit: int,
    ) -> tuple[tuple[RoutingTurn, ...], str | None]:
        if limit <= 0:
            return (), None
        async with self.database.session() as session:
            repository = self._repository(session)
            conversation = await repository.find_by_chat(channel, chat)
            if conversation is None:
                return (), None
            turns = await repository.load_routing_turns(conversation, limit)
            payload = await repository.load_current_summary(conversation)
            summary = payload.get('summary') if payload else None
        return tuple(routing_turn(turn) for turn in turns), (
            summary if isinstance(summary, str) else None
        )

    @override
    async def document_blobs(self, channel: str, chat: str) -> frozenset[str]:
        async with self.database.session() as session, session.begin():
            repository = self._repository(session)
            conversation = await repository.find_by_chat(channel, chat)
            if conversation is None:
                return frozenset()
            return await repository.load_document_blob_ids(conversation)

    @override
    async def record(
        self,
        channel: str,
        chat: str,
        agent: str,
        prompts: Sequence[TurnPrompt],
        answer: str,
    ) -> None:
        async with self._open(channel, chat) as transcript:
            await transcript.record(agent, prompts, answer)

    @override
    async def maintain(
        self,
        channel: str,
        chat: str,
        agent: str,
        ref: ModelRef,
    ) -> None:
        async with self.database.session() as session, session.begin():
            repository = self._repository(session)
            conversation = await repository.get_or_create(channel, chat)
            configured = self.config.section(AGENTS).root[agent]
            service = CompactionService(
                self.llm,
                repository,
                instructions=load_instructions(configured),
                instruction=self.config.section(COMPACTION).summary_prompt,
            )
            result = await service.maybe_compact(conversation, ref)

        if result.compacted:
            logger.info('Compacted conversation %s', conversation.id)

    @asynccontextmanager
    async def _open(
        self,
        channel: str,
        chat: str,
    ) -> AsyncIterator[ConversationTranscript]:
        async with self.database.session() as session, session.begin():
            repository = self._repository(session)
            conversation = await repository.get_or_create(channel, chat)
            yield ConversationTranscript(
                repository,
                self.blobs.for_session(session),
                conversation,
                self.config.section(BLOBS).retention,
            )

    def _repository(self, session: AsyncSession) -> ConversationRepository:
        return ConversationRepository(session, self.guid, self.clock)
