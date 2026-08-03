import asyncio
from typing import override
from uuid import UUID

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.features.conversations.compaction import CompactionService, SummaryOutput
from app.features.conversations.repositories import ConversationRepository
from app.features.conversations.schemas.sqlalchemy import ConversationRecord
from app.features.conversations.types import Conversation, Part, Turn
from lib.guid import GUIDManager
from lib.llm import (
    CompletionChunk,
    FinishChunk,
    LLMClient,
    LLMContract,
    ProviderConfig,
    TextChunk,
    TextSegment,
)
from lib.temporal import ClockManager
from tests.unit.app.features.agents.test_agent_runner import FakeLLM

pytestmark = pytest.mark.integration


class FakeLLMContract(LLMContract):
    client: LLMClient | None

    def __init__(self, client: LLMClient | None) -> None:
        self.client = client

    @override
    def client_for(self, model: str) -> LLMClient | None:
        return self.client

    @override
    def provider_for(self, model: str) -> ProviderConfig | None:
        return None


def _repository(session: AsyncSession) -> ConversationRepository:
    return ConversationRepository(session, GUIDManager(), ClockManager())


def _summary_chunks(text: str) -> list[CompletionChunk]:
    return [TextChunk(text=text), FinishChunk(reason='stop')]


async def _summary_turn_id(
    repository: ConversationRepository,
    conversation: Conversation,
) -> UUID | None:
    turns = await repository.load_recent_turns(conversation, limit=100)
    for turn in turns:
        if turn.parts and turn.parts[0].kind == 'summary':
            return turn.id
    return None


async def test_compact_end_to_end_against_real_postgres(session: AsyncSession) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', 'compact-1', 'scout')
    keep_turns = 4
    turns = [
        await repository.append_turn(
            conversation,
            [Part(role='user', kind='text', payload={'text': f'turn {index}'})],
        )
        for index in range(keep_turns * 2 + 3)
    ]
    llm_client = FakeLLM(
        [
            _summary_chunks(
                '{"summary": "recap", "key_facts": ["fact"], "open_items": []}',
            ),
        ],
    )
    service = CompactionService(
        FakeLLMContract(llm_client),
        repository,
        keep_turns=keep_turns,
    )

    result = await service.maybe_compact(conversation, 'test-model')

    assert result.compacted is True
    assert result.conversation.compact_version == 1
    boundary_index = len(turns) - keep_turns - 1
    assert result.conversation.summarized_through_turn_id == turns[boundary_index].id

    payload = await repository.load_current_summary(result.conversation)
    assert payload is not None
    validated = SummaryOutput.model_validate(payload, strict=False)
    assert validated.summary == 'recap'
    assert validated.key_facts == ('fact',)

    remaining = await repository.load_turns_since_summary(result.conversation)
    assert [turn.id for turn in remaining] == [
        turn.id for turn in turns[boundary_index + 1 :]
    ]


async def test_compact_leaves_database_untouched_when_llm_output_is_not_json(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', 'compact-2', 'scout')
    turns = [
        await repository.append_turn(
            conversation,
            [Part(role='user', kind='text', payload={'text': f'turn {index}'})],
        )
        for index in range(5)
    ]
    llm_client = FakeLLM(
        [
            [TextChunk(text='not json at all'), FinishChunk(reason='stop')],
            [TextChunk(text='still not json'), FinishChunk(reason='stop')],
        ],
    )
    service = CompactionService(
        FakeLLMContract(llm_client),
        repository,
        keep_turns=1,
        max_retries=1,
    )

    result = await service.maybe_compact(conversation, 'test-model')

    assert result.compacted is False
    assert result.conversation.compact_version == 0
    assert result.conversation.summarized_through_turn_id is None
    assert await repository.load_current_summary(conversation) is None

    remaining = await repository.load_turns_since_summary(conversation)
    assert [turn.id for turn in remaining] == [turn.id for turn in turns]
    recent = await repository.load_recent_turns(conversation, limit=10)
    assert [turn.id for turn in recent] == [turn.id for turn in turns]


async def test_second_compaction_folds_prior_summary_turn(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', 'compact-3', 'scout')

    for index in range(3):
        await repository.append_turn(
            conversation,
            [Part(role='user', kind='text', payload={'text': f'first {index}'})],
        )
    first_llm = FakeLLM(
        [
            _summary_chunks(
                '{"summary": "first recap", "key_facts": ["fact-a"],'
                ' "open_items": ["todo-a"]}',
            ),
        ],
    )
    first_service = CompactionService(
        FakeLLMContract(first_llm),
        repository,
        keep_turns=1,
    )
    first_result = await first_service.maybe_compact(conversation, 'test-model')
    assert first_result.compacted is True
    conversation = first_result.conversation
    first_summary_id = await _summary_turn_id(repository, conversation)
    assert first_summary_id is not None

    for index in range(3):
        await repository.append_turn(
            conversation,
            [Part(role='user', kind='text', payload={'text': f'second {index}'})],
        )
    second_llm = FakeLLM(
        [
            _summary_chunks(
                '{"summary": "combined recap",'
                ' "key_facts": ["fact-a", "fact-b"], "open_items": []}',
            ),
        ],
    )
    second_service = CompactionService(
        FakeLLMContract(second_llm),
        repository,
        keep_turns=1,
    )

    second_result = await second_service.maybe_compact(conversation, 'test-model')

    assert second_result.compacted is True
    assert len(second_llm.requests) == 1
    messages = second_llm.requests[0].messages
    # NOTE: the narrative recap opens history (messages[1], right after system) exactly
    # like a normal turn would replay it; the richer key_facts/open_items detail needed
    # to write a new summary lives in the trailing fold instruction instead
    opening = messages[1].segments[0]
    assert isinstance(opening, TextSegment)
    assert 'first recap' in opening.text

    closing = messages[-1].segments[0]
    assert isinstance(closing, TextSegment)
    assert 'fact-a' in closing.text
    assert 'todo-a' in closing.text

    remaining = await repository.load_turns_since_summary(second_result.conversation)
    assert len(remaining) == 1

    # the boundary moved past both the first summary turn and the newly folded
    # real turns, but turns are append-only: the first summary turn is still
    # directly queryable, just no longer part of the active window
    all_recent = await repository.load_recent_turns(
        second_result.conversation,
        limit=50,
    )
    assert first_summary_id in [turn.id for turn in all_recent]


async def test_concurrent_compact_yields_exactly_one_winner(database_url: str) -> None:
    engine_setup = create_async_engine(database_url)
    session_setup = AsyncSession(engine_setup)
    try:
        setup_repository = _repository(session_setup)
        conversation = await setup_repository.get_or_create(
            'telegram',
            'compact-race',
            'scout',
        )
        turns = tuple(
            [
                await setup_repository.append_turn(
                    conversation,
                    [Part(role='user', kind='text', payload={'text': f'turn {index}'})],
                )
                for index in range(5)
            ],
        )
        await session_setup.commit()
    finally:
        await session_setup.close()
        await engine_setup.dispose()

    engine_a = create_async_engine(database_url)
    engine_b = create_async_engine(database_url)
    session_a = AsyncSession(engine_a)
    session_b = AsyncSession(engine_b)

    try:
        repository_a = _repository(session_a)
        repository_b = _repository(session_b)
        conversation_a = await repository_a.get_or_create(
            'telegram',
            'compact-race',
            'scout',
        )
        conversation_b = await repository_b.get_or_create(
            'telegram',
            'compact-race',
            'scout',
        )
        assert conversation_a.compact_version == 0
        assert conversation_b.compact_version == 0

        async def race(
            service: CompactionService,
            conversation: Conversation,
            turns_to_fold: tuple[Turn, ...],
            session: AsyncSession,
        ) -> bool:
            result = await service.compact(
                conversation,
                None,
                turns_to_fold,
                'test-model',
            )
            # NOTE: commit inside the racing coroutine itself, not after gather() —
            # the loser's UPDATE blocks on the winner's row lock at the database
            # level; if neither side committed until after both coroutines
            # returned, that would deadlock (the loser would wait forever for a
            # commit that never comes until it itself returns).
            await session.commit()
            return result.compacted

        service_a = CompactionService(
            FakeLLMContract(FakeLLM([_summary_chunks('{"summary": "race a"}')])),
            repository_a,
        )
        service_b = CompactionService(
            FakeLLMContract(FakeLLM([_summary_chunks('{"summary": "race b"}')])),
            repository_b,
        )

        outcomes = await asyncio.gather(
            race(service_a, conversation_a, turns[:-1], session_a),
            race(service_b, conversation_b, turns[:-1], session_b),
        )

        assert sorted(outcomes) == [False, True]
    finally:
        await session_a.close()
        await session_b.close()
        await engine_a.dispose()
        await engine_b.dispose()

    engine_verify = create_async_engine(database_url)
    session_verify = AsyncSession(engine_verify)
    try:
        repository_verify = _repository(session_verify)
        final = await repository_verify.get_or_create(
            'telegram',
            'compact-race',
            'scout',
        )
        assert final.compact_version == 1

        recent = await repository_verify.load_recent_turns(final, limit=50)
        summary_turns = [turn for turn in recent if turn.parts[0].kind == 'summary']
        assert len(summary_turns) == 1
    finally:
        await session_verify.execute(
            sa.delete(ConversationRecord).where(ConversationRecord.id == final.id),
        )
        await session_verify.commit()
        await session_verify.close()
        await engine_verify.dispose()
