from collections.abc import AsyncIterator
from typing import override

import pytest

from app.core.config.registry import HuginnConfig
from app.features.agents.config import AgentConfig, AgentConfigList, DefaultsConfig
from app.features.conversations.repositories import (
    BlobRepository,
    ConversationRepository,
)
from app.features.conversations.services import (
    ConversationControl,
    ConversationTranscript,
)
from app.features.conversations.types import Part
from lib.database import DatabaseContract, DatabaseManager
from lib.guid import GUIDManager
from lib.llm import (
    CompletionChunk,
    CompletionRequest,
    FinishChunk,
    LLMClient,
    LLMContract,
    ProviderConfig,
    TextChunk,
    TextSegment,
)
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration


class Summarizer(LLMClient):
    calls: int

    def __init__(self) -> None:
        self.calls = 0

    @override
    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[CompletionChunk]:
        self.calls += 1
        yield TextChunk(text='{"summary": "they said things"}')
        yield FinishChunk(reason='stop')


class Models(LLMContract):
    client: Summarizer

    def __init__(self, client: Summarizer) -> None:
        self.client = client

    @override
    def client_for(self, model: str) -> LLMClient | None:
        return self.client

    @override
    def provider_for(self, model: str) -> ProviderConfig | None:
        return None


def _config() -> HuginnConfig:
    return HuginnConfig(
        sections={
            'agents': AgentConfigList(
                {
                    'scout': AgentConfig(description='scouts', instructions='be brief'),
                    'sage': AgentConfig(description='sages', instructions='be brief'),
                },
            ),
            'defaults': DefaultsConfig(model='test-model'),
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
        conversation = await repository.get_or_create('support_bot', chat, agent)
        await repository.append_turn(
            conversation,
            [
                Part(role='user', kind='text', payload={'text': f'hello {agent}'}),
                Part(role='assistant', kind='text', payload={'text': 'hi'}),
            ],
        )


async def _history(database: DatabaseContract, chat: str, agent: str) -> list[str]:
    async with database.session() as session, session.begin():
        repository = ConversationRepository(session, GUIDManager(), ClockManager())
        conversation = await repository.get_or_create('support_bot', chat, agent)
        transcript = ConversationTranscript(
            repository,
            BlobRepository(session, GUIDManager()),
            conversation,
        )

        return [
            segment.text
            for message in await transcript.history()
            for segment in message.segments
            if isinstance(segment, TextSegment)
        ]


@pytest.mark.usefixtures('_migrated')
async def test_reset_clears_every_agent_of_the_chat_and_leaves_others(
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

        # NOTE: Han's ruling -- a chat-level command means the whole chat, so every
        # agent's conversation in it starts over and no other chat is touched
        assert cleared == 2
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
        assert await _control(database, Models(llm)).compact('support_bot', '902') == 1
        assert await _history(database, '902', 'scout') == [
            'Summary of the conversation so far: they said things',
        ]

        assert await _control(database, Models(llm)).reset('support_bot', '902') == 1

        # NOTE: the summary lives in a turn after the boundary compaction set, so the
        # reset has to cut it off too or the model keeps being told the old story
        assert await _history(database, '902', 'scout') == []
    finally:
        await database.dispose()


@pytest.mark.usefixtures('_migrated')
async def test_compact_folds_every_agent_of_the_chat_once(
    database_url: str,
) -> None:
    database = DatabaseManager({None: database_url})
    llm = Summarizer()
    try:
        await _said(database, '903', 'scout')
        await _said(database, '903', 'sage')

        folded = await _control(database, Models(llm)).compact('support_bot', '903')

        assert folded == 2
        assert llm.calls == 2
        assert await _history(database, '903', 'scout') == [
            'Summary of the conversation so far: they said things',
        ]
    finally:
        await database.dispose()


@pytest.mark.usefixtures('_migrated')
async def test_nothing_to_fold_asks_no_model(database_url: str) -> None:
    database = DatabaseManager({None: database_url})
    llm = Summarizer()
    try:
        await _said(database, '904', 'scout')
        assert await _control(database, Models(llm)).reset('support_bot', '904') == 1

        folded = await _control(database, Models(llm)).compact('support_bot', '904')

        assert folded == 0
        assert llm.calls == 0
    finally:
        await database.dispose()
