import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import override

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config.registry import HuginnConfig
from app.features.agents.config import (
    AGENTS,
    DEFAULTS,
    load_instructions,
    resolve_model,
)
from app.features.agents.contracts import TranscriptStoreContract
from app.features.agents.types import PromptAttachment, TurnPrompt
from lib.database import DatabaseContract
from lib.guid import GUIDContract
from lib.llm import LLMContract, Message
from lib.temporal import ClockContract

from .compaction import (
    CompactionService,
    SummaryOutput,
    replay_messages,
    summary_message,
)
from .config import COMPACTION
from .contracts import (
    BlobStoreContract,
    BlobStoreFactory,
    ConversationControlContract,
)
from .repositories import ConversationRepository
from .types import Conversation, Part

_TEXT = 'text'

logger = logging.getLogger(__name__)


class ConversationTranscript:
    blobs: BlobStoreContract
    conversation: Conversation
    repository: ConversationRepository

    def __init__(
        self,
        repository: ConversationRepository,
        blobs: BlobStoreContract,
        conversation: Conversation,
    ) -> None:
        self.blobs = blobs
        self.conversation = conversation
        self.repository = repository

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

    async def record(self, prompts: Sequence[TurnPrompt], answer: str) -> None:
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

        await self.repository.append_turn(self.conversation, parts)

    async def _media(self, attachment: PromptAttachment, actor: str) -> Part:
        blob = await self.blobs.store(attachment.mime, attachment.data)

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
    async def compact(self, channel: str, chat: str) -> int:
        compacted = 0
        async with self.database.session() as session, session.begin():
            repository = ConversationRepository(session, self.guid, self.clock)
            for conversation in await repository.list_by_chat(channel, chat):
                configured = self.config.section(AGENTS).root.get(conversation.agent)
                if configured is None:
                    logger.warning(
                        'Conversation %s names agent %s, which is not configured',
                        conversation.id,
                        conversation.agent,
                    )
                    continue

                model = resolve_model(
                    configured.model,
                    self.config.section(DEFAULTS).model,
                )
                service = CompactionService(
                    self.llm,
                    repository,
                    instructions=load_instructions(configured),
                    instruction=self.config.section(COMPACTION).summary_prompt,
                )
                result = await service.force(conversation, model)
                compacted += int(result.compacted)

        return compacted

    @override
    async def reset(self, channel: str, chat: str) -> int:
        async with self.database.session() as session, session.begin():
            repository = ConversationRepository(session, self.guid, self.clock)
            reset = [
                await repository.reset(conversation, conversation.compact_version)
                for conversation in await repository.list_by_chat(channel, chat)
            ]

        return sum(1 for conversation in reset if conversation is not None)


class ConversationTranscriptStore(TranscriptStoreContract):
    blobs: BlobStoreFactory
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
        blobs: BlobStoreFactory,
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
        agent: str,
    ) -> tuple[Message, ...]:
        async with self._open(channel, chat, agent) as transcript:
            return await transcript.history()

    @override
    async def record(
        self,
        channel: str,
        chat: str,
        agent: str,
        prompts: Sequence[TurnPrompt],
        answer: str,
    ) -> None:
        async with self._open(channel, chat, agent) as transcript:
            await transcript.record(prompts, answer)

    @override
    async def maintain(
        self,
        channel: str,
        chat: str,
        agent: str,
        model: str,
    ) -> None:
        async with self.database.session() as session, session.begin():
            repository = self._repository(session)
            conversation = await repository.get_or_create(channel, chat, agent)
            configured = self.config.section(AGENTS).root[agent]
            service = CompactionService(
                self.llm,
                repository,
                instructions=load_instructions(configured),
                instruction=self.config.section(COMPACTION).summary_prompt,
            )
            result = await service.maybe_compact(conversation, model)

        if result.compacted:
            logger.info('Compacted conversation %s', conversation.id)

    @asynccontextmanager
    async def _open(
        self,
        channel: str,
        chat: str,
        agent: str,
    ) -> AsyncIterator[ConversationTranscript]:
        async with self.database.session() as session, session.begin():
            repository = self._repository(session)
            conversation = await repository.get_or_create(channel, chat, agent)
            yield ConversationTranscript(
                repository,
                self.blobs(session),
                conversation,
            )

    def _repository(self, session: AsyncSession) -> ConversationRepository:
        return ConversationRepository(session, self.guid, self.clock)
