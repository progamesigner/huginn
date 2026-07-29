from collections.abc import Sequence
from typing import override

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.agents.types import TurnPrompt
from app.features.conversations.repositories import ConversationRepository
from app.features.conversations.services import (
    ConversationTranscript,
    ConversationTranscriptStore,
)
from app.features.conversations.types import Part
from lib.database import DatabaseContract, DatabaseManager
from lib.guid import GUIDManager
from lib.llm import (
    LLMClient,
    LLMContract,
    Message,
    ProviderConfig,
    TextSegment,
)
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration


class NoModels(LLMContract):
    @override
    def client_for(self, model: str) -> LLMClient | None:
        return None

    @override
    def provider_for(self, model: str) -> ProviderConfig | None:
        return None


def _repository(session: AsyncSession) -> ConversationRepository:
    return ConversationRepository(session, GUIDManager(), ClockManager())


def _store(database: DatabaseContract) -> ConversationTranscriptStore:
    return ConversationTranscriptStore(
        database,
        GUIDManager(),
        ClockManager(),
        NoModels(),
    )


def _prompt(text: str) -> TurnPrompt:
    return TurnPrompt(text=text, actor='user:1')


def _texts(messages: Sequence[Message]) -> list[tuple[str, str]]:
    return [
        (message.role, segment.text)
        for message in messages
        for segment in message.segments
        if isinstance(segment, TextSegment)
    ]


async def test_a_recorded_turn_reads_back_as_the_next_turns_history(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '4242', 'scout')
    transcript = ConversationTranscript(repository, conversation)

    await transcript.record([_prompt('first'), _prompt('second')], 'answer')

    assert _texts(await transcript.history()) == [
        ('user', 'first'),
        ('user', 'second'),
        ('assistant', 'answer'),
    ]


async def test_an_empty_answer_records_only_what_the_user_said(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '1', 'scout')
    transcript = ConversationTranscript(repository, conversation)

    await transcript.record([_prompt('hello')], '')

    assert _texts(await transcript.history()) == [('user', 'hello')]


async def test_history_opens_with_the_summary_and_omits_folded_turns(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '2', 'scout')
    transcript = ConversationTranscript(repository, conversation)
    await transcript.record([_prompt('ancient')], 'forgotten')
    folded = await repository.load_turns_since_summary(conversation)
    updated = await repository.apply_compaction(
        conversation,
        summary_payload={'summary': 'they discussed the ancients'},
        turns_folded=folded,
        expected_version=conversation.compact_version,
    )

    assert updated is not None
    after = ConversationTranscript(repository, updated)
    await after.record([_prompt('recent')], 'fresh')

    # NOTE: the summary stands in for everything before the boundary, so the folded turn
    # must not come back as well -- that would send the model the same content twice
    assert _texts(await after.history()) == [
        ('user', 'Summary of the conversation so far: they discussed the ancients'),
        ('user', 'recent'),
        ('assistant', 'fresh'),
    ]


async def test_tool_traffic_is_kept_but_never_replayed(session: AsyncSession) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '3', 'scout')
    await repository.append_turn(
        conversation,
        [
            Part(role='user', kind='text', payload={'text': 'search it'}),
            Part(role='assistant', kind='tool_call', payload={'name': 'websearch'}),
            Part(role='tool', kind='tool_result', payload={'text': 'raw hits'}),
            Part(role='assistant', kind='text', payload={'text': 'here you go'}),
        ],
    )

    assert _texts(await ConversationTranscript(repository, conversation).history()) == [
        ('user', 'search it'),
        ('assistant', 'here you go'),
    ]


@pytest.mark.usefixtures('_migrated')
async def test_the_store_opens_a_transcript_per_round_and_commits_it(
    database_url: str,
) -> None:
    database = DatabaseManager({None: database_url})
    store = _store(database)
    try:
        async with store.open('support_bot', '7777', 'scout') as transcript:
            await transcript.record([_prompt('hello')], 'answer')

        # NOTE: a second round is a second session, so seeing the first round's turn
        # proves the round committed rather than merely flushed
        async with store.open('support_bot', '7777', 'scout') as transcript:
            assert _texts(await transcript.history()) == [
                ('user', 'hello'),
                ('assistant', 'answer'),
            ]
    finally:
        await database.dispose()


async def _failing_round(store: ConversationTranscriptStore) -> None:
    async with store.open('support_bot', '8888', 'scout') as transcript:
        await transcript.record([_prompt('hello')], 'answer')
        raise RuntimeError('llm stream died')


@pytest.mark.usefixtures('_migrated')
async def test_a_round_that_raises_commits_nothing(database_url: str) -> None:
    database = DatabaseManager({None: database_url})
    store = _store(database)
    try:
        with pytest.raises(RuntimeError):
            await _failing_round(store)

        async with store.open('support_bot', '8888', 'scout') as transcript:
            assert await transcript.history() == ()
    finally:
        await database.dispose()
