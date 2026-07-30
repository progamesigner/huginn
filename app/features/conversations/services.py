import logging
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import override

from pydantic import JsonValue
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config.registry import HuginnConfig
from app.features.agents.config import AGENTS, DEFAULTS, resolve_model
from app.features.agents.contracts import (
    TranscriptContract,
    TranscriptStoreContract,
)
from app.features.agents.types import PromptAttachment, TurnPrompt
from lib.database import DatabaseContract
from lib.guid import GUIDContract
from lib.llm import LLMContract, Message, TextSegment
from lib.temporal import ClockContract

from .compaction import CompactionService
from .contracts import (
    BlobStoreContract,
    BlobStoreFactory,
    ConversationControlContract,
)
from .repositories import ConversationRepository
from .types import Conversation, Part, Turn

logger = logging.getLogger(__name__)

_TEXT = 'text'
_REPLAYED = ('audio', 'image', _TEXT)


def _messages(turns: Sequence[Turn]) -> Iterator[Message]:
    for turn in turns:
        for part in turn.parts:
            # NOTE: role is a constrained column, not a literal type, so replaying it
            # into a model turn has to narrow it back to the two roles that can speak
            if (text := _text_of(part)) is None:
                continue

            if part.role == 'assistant':
                yield Message(role='assistant', segments=(TextSegment(text=text),))
            elif part.role == 'user':
                yield Message(role='user', segments=(TextSegment(text=text),))


def _summary_message(summary: Mapping[str, JsonValue]) -> Message | None:
    text = summary.get('summary')
    if not isinstance(text, str) or not text:
        return None

    return Message(
        role='user',
        segments=(TextSegment(text=f'Summary of the conversation so far: {text}'),),
        cacheable=True,
    )


def _text_of(part: Part) -> str | None:
    if part.kind not in _REPLAYED:
        return None

    text = part.payload.get('text')
    return text if isinstance(text, str) and text else None


class ConversationTranscript(TranscriptContract):
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

    @override
    async def history(self) -> tuple[Message, ...]:
        summary = await self.repository.load_current_summary(self.conversation)
        turns = await self.repository.load_turns_since_summary(self.conversation)
        opening = None if summary is None else _summary_message(summary)

        return (
            *(() if opening is None else (opening,)),
            *_messages(turns),
        )

    @override
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
            service = CompactionService(self.llm, repository)
            for conversation in await repository.list_by_chat(channel, chat):
                model = self._model(conversation.agent)
                if model is None:
                    logger.warning(
                        'Conversation %s names agent %s, which is not configured',
                        conversation.id,
                        conversation.agent,
                    )
                    continue

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

    def _model(self, agent: str) -> str | None:
        configured = self.config.section(AGENTS).root.get(agent)
        if configured is None:
            return None

        return resolve_model(configured.model, self.config.section(DEFAULTS).model)


class ConversationTranscriptStore(TranscriptStoreContract):
    blobs: BlobStoreFactory
    clock: ClockContract
    database: DatabaseContract
    guid: GUIDContract
    llm: LLMContract

    def __init__(
        self,
        database: DatabaseContract,
        guid: GUIDContract,
        clock: ClockContract,
        llm: LLMContract,
        blobs: BlobStoreFactory,
    ) -> None:
        self.blobs = blobs
        self.clock = clock
        self.database = database
        self.guid = guid
        self.llm = llm

    @override
    def open(
        self,
        channel: str,
        chat: str,
        agent: str,
    ) -> AbstractAsyncContextManager[TranscriptContract]:
        return self._open(channel, chat, agent)

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
            result = await CompactionService(self.llm, repository).maybe_compact(
                conversation,
                model,
            )

        if result.compacted:
            logger.info('Compacted conversation %s', conversation.id)

    @asynccontextmanager
    async def _open(
        self,
        channel: str,
        chat: str,
        agent: str,
    ) -> AsyncIterator[TranscriptContract]:
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
