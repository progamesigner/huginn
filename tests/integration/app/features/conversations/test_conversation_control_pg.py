from collections.abc import AsyncIterator
from typing import override

import pytest

from app.core.config.registry import HuginnConfig
from app.features.agents.config import AgentConfig, AgentConfigList, DefaultsConfig
from app.features.blobs.config import RetentionConfig
from app.features.blobs.services import BlobService
from app.features.conversations.config import CompactionConfig
from app.features.conversations.entities import Part
from app.features.conversations.repositories import ConversationRepository
from app.features.conversations.services import (
    ConversationControl,
    ConversationTranscript,
)
from lib.database import DatabaseContract, DatabaseManager
from lib.guid import GUIDManager
from lib.llm import (
    CompletionChunk,
    CompletionRequest,
    FinishChunk,
    LLMClient,
    LLMContract,
    ModelRef,
    ProviderConfig,
    TextChunk,
    TextSegment,
)
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration


class Summarizer(LLMClient):
    calls: int
    requests: list[CompletionRequest]

    def __init__(self) -> None:
        self.calls = 0
        self.requests = []

    @override
    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[CompletionChunk]:
        self.calls += 1
        self.requests.append(request)
        yield TextChunk(text='{"summary": "they said things"}')
        yield FinishChunk(reason='stop')


class Models(LLMContract):
    client: Summarizer

    def __init__(self, client: Summarizer) -> None:
        self.client = client

    @override
    def client_for(self, ref: ModelRef) -> LLMClient | None:
        return self.client

    @override
    def provider_for(self, ref: ModelRef) -> ProviderConfig | None:
        return None


def _config() -> HuginnConfig:
    return HuginnConfig(
        sections={
            'agents': AgentConfigList(
                {
                    'scout': AgentConfig(description='scouts', instructions='be brief'),
                    'sage': AgentConfig(
                        description='sages',
                        instructions='be thorough',
                    ),
                },
            ),
            'defaults': DefaultsConfig(model='test-model'),
            'compaction': CompactionConfig(),
        },
    )


def _control(database: DatabaseContract, llm: LLMContract) -> ConversationControl:
    return ConversationControl(
        database,
        GUIDManager(),
        ClockManager(),
        llm,
        _config(),
    )


async def _said(database: DatabaseContract, chat: str, agent: str) -> None:
    async with database.session() as session, session.begin():
        repository = ConversationRepository(session, GUIDManager(), ClockManager())
        conversation = await repository.get_or_create('support_bot', chat)
        await repository.append_turn(
            conversation,
            agent,
            [
                Part(role='user', kind='text', payload={'text': f'hello {agent}'}),
                Part(role='assistant', kind='text', payload={'text': 'hi'}),
            ],
        )


async def _history(database: DatabaseContract, chat: str, agent: str) -> list[str]:
    async with database.transaction() as session:
        repository = ConversationRepository(session, GUIDManager(), ClockManager())
        conversation = await repository.get_or_create('support_bot', chat)
        transcript = ConversationTranscript(
            session,
            repository,
            BlobService(database, GUIDManager(), ClockManager()),
            conversation,
            RetentionConfig(),
        )

        return [
            segment.text
            for message in await transcript.history()
            for segment in message.segments
            if isinstance(segment, TextSegment)
        ]


@pytest.mark.usefixtures('_migrated')
async def test_reset_clears_the_shared_chat_and_leaves_others(
    database_url: str,
) -> None:
    database = DatabaseManager({None: database_url})
    try:
        await _said(database, '900', 'scout')
        await _said(database, '900', 'sage')
        await _said(database, '901', 'scout')

        cleared = await _control(database, Models(Summarizer())).reset(
            'support_bot',
            '900',
        )

        assert cleared is True
        assert await _history(database, '900', 'scout') == []
        assert await _history(database, '900', 'sage') == []
        assert await _history(database, '901', 'scout') == ['hello scout', 'hi']
    finally:
        await database.dispose()


@pytest.mark.usefixtures('_migrated')
async def test_a_reset_chat_does_not_replay_the_summary_it_had_before(
    database_url: str,
) -> None:
    database = DatabaseManager({None: database_url})
    llm = Summarizer()
    try:
        await _said(database, '902', 'scout')
        assert (
            await _control(database, Models(llm)).compact('support_bot', '902') is True
        )
        assert await _history(database, '902', 'scout') == [
            'Summary of the conversation so far: they said things',
        ]

        assert await _control(database, Models(llm)).reset('support_bot', '902') is True

        # NOTE: the summary lives in a turn after the boundary compaction set, so the
        # reset has to cut it off too or the model keeps being told the old story
        assert await _history(database, '902', 'scout') == []
    finally:
        await database.dispose()


@pytest.mark.usefixtures('_migrated')
async def test_compact_folds_the_shared_chat_once(
    database_url: str,
) -> None:
    database = DatabaseManager({None: database_url})
    llm = Summarizer()
    try:
        await _said(database, '903', 'scout')
        await _said(database, '903', 'sage')

        folded = await _control(database, Models(llm)).compact('support_bot', '903')

        assert folded is True
        assert llm.calls == 1
        assert await _history(database, '903', 'scout') == [
            'Summary of the conversation so far: they said things',
        ]
    finally:
        await database.dispose()


@pytest.mark.usefixtures('_migrated')
async def test_compact_uses_the_latest_agents_instructions(
    database_url: str,
) -> None:
    database = DatabaseManager({None: database_url})
    llm = Summarizer()
    try:
        await _said(database, '905', 'scout')
        await _said(database, '905', 'sage')

        await _control(database, Models(llm)).compact('support_bot', '905')

        assert all(request.messages[0].role == 'system' for request in llm.requests)
        texts = [
            segment.text
            for request in llm.requests
            for segment in [request.messages[0].segments[0]]
            if isinstance(segment, TextSegment)
        ]
        assert texts == ['be thorough']
    finally:
        await database.dispose()


@pytest.mark.usefixtures('_migrated')
async def test_nothing_to_fold_asks_no_model(database_url: str) -> None:
    database = DatabaseManager({None: database_url})
    llm = Summarizer()
    try:
        await _said(database, '904', 'scout')
        assert await _control(database, Models(llm)).reset('support_bot', '904') is True

        folded = await _control(database, Models(llm)).compact('support_bot', '904')

        assert folded is False
        assert llm.calls == 0
    finally:
        await database.dispose()


@pytest.mark.usefixtures('_migrated')
async def test_compact_skips_an_empty_conversation(database_url: str) -> None:
    database = DatabaseManager({None: database_url})
    llm = Summarizer()
    try:
        async with database.session() as session, session.begin():
            repository = ConversationRepository(session, GUIDManager(), ClockManager())
            await repository.get_or_create('support_bot', 'empty')

        assert (
            await _control(database, Models(llm)).compact('support_bot', 'empty')
            is False
        )
        assert llm.calls == 0
    finally:
        await database.dispose()


@pytest.mark.usefixtures('_migrated')
async def test_compact_skips_a_removed_latest_agent(
    database_url: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = DatabaseManager({None: database_url})
    llm = Summarizer()
    try:
        await _said(database, 'retired', 'retired-agent')

        with caplog.at_level('WARNING'):
            result = await _control(database, Models(llm)).compact(
                'support_bot',
                'retired',
            )

        assert result is False
        assert llm.calls == 0
        assert any('retired-agent' in message for message in caplog.messages)
    finally:
        await database.dispose()
