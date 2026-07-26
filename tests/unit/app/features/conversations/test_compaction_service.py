from typing import cast, override
from uuid import UUID

import pytest

from app.features.conversations.compaction import CompactionService
from app.features.conversations.exceptions import NoTurnsToCompactError
from app.features.conversations.repositories import ConversationRepository
from app.features.conversations.types import Part, Turn
from lib.llm import (
    FinishChunk,
    LLMClient,
    LLMContract,
    Message,
    ProviderConfig,
    TextChunk,
    TextSegment,
)
from tests.unit.app.features.agents.test_agent_runner import FakeLLM
from tests.unit.app.features.conversations.test_conversation_repository import (
    FakeConversationRepository,
)


class FakeLLMContract(LLMContract):
    client: LLMClient | None
    calls: int

    def __init__(self, client: LLMClient | None) -> None:
        self.client = client
        self.calls = 0

    @override
    def client_for(self, model: str) -> LLMClient | None:
        self.calls += 1
        return self.client

    @override
    def provider_for(self, model: str) -> ProviderConfig | None:
        return None


def prompt_text(message: Message) -> str:
    segment = message.segments[0]
    assert isinstance(segment, TextSegment)
    return segment.text


def build_service(
    llm: LLMContract,
    repository: FakeConversationRepository,
    **kwargs: int,
) -> CompactionService:
    # NOTE: FakeConversationRepository duck-types ConversationRepository's surface;
    # cast is type-only, matching the established fake-injection pattern (e.g.
    # ScopedSession(cast(AsyncSession, FakeSession())) in
    # tests/unit/lib/database/test_database_manager.py).
    return CompactionService(llm, cast(ConversationRepository, repository), **kwargs)


async def test_compact_persists_summary_turn_and_advances_boundary() -> None:
    repository = FakeConversationRepository()
    conversation = await repository.get_or_create('telegram', '1', 'scout')
    turns = [
        await repository.append_turn(
            conversation,
            [Part(role='user', kind='text', payload={'text': f'turn {index}'})],
        )
        for index in range(3)
    ]
    llm_client = FakeLLM(
        [
            [
                TextChunk(
                    text='{"summary": "recap", "key_facts": [], "open_items": []}',
                ),
                FinishChunk(reason='stop'),
            ],
        ],
    )
    service = build_service(FakeLLMContract(llm_client), repository)

    result = await service.compact(conversation, None, tuple(turns), 'test-model')

    assert result.compacted is True
    assert result.conversation.compact_version == 1
    assert result.conversation.summarized_through_turn_id == turns[-1].id

    summary = await repository.load_current_summary(result.conversation)
    assert summary == {'summary': 'recap', 'key_facts': [], 'open_items': []}
    assert await repository.load_turns_since_summary(result.conversation) == ()

    assert len(llm_client.requests) == 1
    output_schema = llm_client.requests[0].output_schema
    assert output_schema is not None
    assert output_schema.name == 'SummaryOutput'
    required = cast(list[str], output_schema.json_schema['required'])
    assert set(required) >= {'key_facts', 'open_items'}


async def test_compact_abandons_on_malformed_llm_output() -> None:
    repository = FakeConversationRepository()
    conversation = await repository.get_or_create('telegram', '2', 'scout')
    turn = await repository.append_turn(
        conversation,
        [Part(role='user', kind='text', payload={'text': 'hi'})],
    )
    non_json = [
        TextChunk(text='Sure, here is a summary: the user asked about X.'),
        FinishChunk(reason='stop'),
    ]
    llm_client = FakeLLM([non_json, non_json])
    service = build_service(FakeLLMContract(llm_client), repository, max_retries=1)

    result = await service.compact(conversation, None, (turn,), 'test-model')

    assert result.compacted is False
    assert result.conversation == conversation
    assert len(llm_client.requests) == 2
    assert await repository.load_current_summary(conversation) is None
    assert await repository.load_turns_since_summary(conversation) == (turn,)


@pytest.mark.parametrize(
    'raw',
    [
        '{"content": "wrong key entirely"}',
        '{"summary": 123}',
    ],
)
async def test_compact_abandons_on_well_formed_json_with_invalid_shape(
    raw: str,
) -> None:
    repository = FakeConversationRepository()
    conversation = await repository.get_or_create('telegram', '3', 'scout')
    turn = await repository.append_turn(
        conversation,
        [Part(role='user', kind='text', payload={'text': 'hi'})],
    )
    chunks = [TextChunk(text=raw), FinishChunk(reason='stop')]
    llm_client = FakeLLM([chunks, chunks])
    service = build_service(FakeLLMContract(llm_client), repository, max_retries=1)

    result = await service.compact(conversation, None, (turn,), 'test-model')

    assert result.compacted is False
    assert result.conversation == conversation
    assert len(llm_client.requests) == 2


async def test_compact_abandons_on_cas_conflict_without_retrying_the_llm() -> None:
    repository = FakeConversationRepository()
    conversation = await repository.get_or_create('telegram', '4', 'scout')
    turn = await repository.append_turn(
        conversation,
        [Part(role='user', kind='text', payload={'text': 'hi'})],
    )
    repository.force_cas_conflict = True
    llm_client = FakeLLM(
        [
            [
                TextChunk(text='{"summary": "ok", "key_facts": [], "open_items": []}'),
                FinishChunk(reason='stop'),
            ],
        ],
    )
    service = build_service(FakeLLMContract(llm_client), repository)

    result = await service.compact(conversation, None, (turn,), 'test-model')

    assert result.compacted is False
    assert result.conversation == conversation
    assert len(llm_client.requests) == 1
    assert await repository.load_current_summary(conversation) is None


@pytest.mark.parametrize(
    ('turn_count', 'payload_size', 'keep_turns', 'token_threshold', 'expected'),
    [
        (2, 10, 5, 50_000, False),
        (5, 10, 5, 50_000, True),
        (1, 400_000, 5, 50_000, True),
        (5, 400_000, 5, 50_000, True),
        (0, 0, 5, 50_000, False),
    ],
    ids=[
        'neither_signal_crossed',
        'turn_count_alone_crossed',
        'token_estimate_alone_crossed',
        'both_crossed',
        'empty_foldable',
    ],
)
def test_should_compact_combined_trigger_table(
    *,
    turn_count: int,
    payload_size: int,
    keep_turns: int,
    token_threshold: int,
    expected: bool,
) -> None:
    repository = FakeConversationRepository()
    service = build_service(
        FakeLLMContract(None),
        repository,
        keep_turns=keep_turns,
        token_threshold=token_threshold,
    )
    foldable = tuple(
        Turn(
            id=UUID(int=index + 1),
            parts=(
                Part(role='user', kind='text', payload={'text': 'x' * payload_size}),
            ),
        )
        for index in range(turn_count)
    )

    assert service.should_compact(foldable) is expected


async def test_maybe_compact_never_calls_the_llm_while_within_keep_turns() -> None:
    repository = FakeConversationRepository()
    conversation = await repository.get_or_create('telegram', '6', 'scout')
    keep_turns = 3
    for _ in range(keep_turns):
        await repository.append_turn(
            conversation,
            [Part(role='user', kind='text', payload={'text': 'x' * 1_000_000})],
        )
    llm_contract = FakeLLMContract(None)
    service = build_service(
        llm_contract,
        repository,
        keep_turns=keep_turns,
        token_threshold=100,
    )

    result = await service.maybe_compact(conversation, 'test-model')

    assert result.compacted is False
    assert llm_contract.calls == 0


async def test_maybe_compact_folds_prior_summary_and_accumulates_context() -> None:
    repository = FakeConversationRepository()
    conversation = await repository.get_or_create('telegram', '7', 'scout')

    for index in range(3):
        await repository.append_turn(
            conversation,
            [Part(role='user', kind='text', payload={'text': f'first {index}'})],
        )
    first_llm = FakeLLM(
        [
            [
                TextChunk(
                    text=(
                        '{"summary": "first recap", "key_facts": ["fact-a"],'
                        ' "open_items": ["todo-a"]}'
                    ),
                ),
                FinishChunk(reason='stop'),
            ],
        ],
    )
    first_service = build_service(
        FakeLLMContract(first_llm),
        repository,
        keep_turns=1,
        token_threshold=50_000,
    )
    first_result = await first_service.maybe_compact(conversation, 'test-model')
    assert first_result.compacted is True
    conversation = first_result.conversation

    for index in range(3):
        await repository.append_turn(
            conversation,
            [Part(role='user', kind='text', payload={'text': f'second {index}'})],
        )
    second_llm = FakeLLM(
        [
            [
                TextChunk(
                    text=(
                        '{"summary": "combined recap",'
                        ' "key_facts": ["fact-a", "fact-b"], "open_items": []}'
                    ),
                ),
                FinishChunk(reason='stop'),
            ],
        ],
    )
    second_service = build_service(
        FakeLLMContract(second_llm),
        repository,
        keep_turns=1,
        token_threshold=50_000,
    )

    second_result = await second_service.maybe_compact(conversation, 'test-model')

    assert second_result.compacted is True
    assert len(second_llm.requests) == 1
    prompt = prompt_text(second_llm.requests[0].messages[0])
    assert 'first recap' in prompt
    assert 'fact-a' in prompt
    assert 'todo-a' in prompt

    remaining = await repository.load_turns_since_summary(second_result.conversation)
    assert len(remaining) == 1

    summary = await repository.load_current_summary(second_result.conversation)
    assert summary is not None
    assert summary['summary'] == 'combined recap'


async def test_summary_request_leads_with_cacheable_transcript_then_instruction() -> (
    None
):
    repository = FakeConversationRepository()
    conversation = await repository.get_or_create('telegram', '10', 'scout')
    turn = await repository.append_turn(
        conversation,
        [Part(role='user', kind='text', payload={'text': 'the timezone is UTC+8'})],
    )
    llm_client = FakeLLM(
        [
            [
                TextChunk(text='{"summary": "ok", "key_facts": [], "open_items": []}'),
                FinishChunk(reason='stop'),
            ],
        ],
    )
    service = build_service(FakeLLMContract(llm_client), repository)

    await service.compact(conversation, None, (turn,), 'test-model')

    messages = llm_client.requests[0].messages
    # NOTE: transcript leads as a cacheable prefix, instruction trails — the prompt
    # ordering that lets provider caches be reused (see _summary_messages).
    assert len(messages) == 2
    assert messages[0].cacheable is True
    assert 'UTC+8' in prompt_text(messages[0])
    assert messages[1].cacheable is False
    assert 'JSON object' in prompt_text(messages[1])


async def test_compact_returns_uncompacted_when_no_client_for_model() -> None:
    repository = FakeConversationRepository()
    conversation = await repository.get_or_create('telegram', '8', 'scout')
    turn = await repository.append_turn(
        conversation,
        [Part(role='user', kind='text', payload={'text': 'hi'})],
    )
    service = build_service(FakeLLMContract(None), repository)

    result = await service.compact(conversation, None, (turn,), 'unknown-model')

    assert result.compacted is False
    assert result.conversation == conversation


async def test_compact_with_no_turns_to_fold_raises() -> None:
    repository = FakeConversationRepository()
    conversation = await repository.get_or_create('telegram', '9', 'scout')
    service = build_service(FakeLLMContract(None), repository)

    with pytest.raises(NoTurnsToCompactError):
        await service.compact(conversation, None, (), 'test-model')
