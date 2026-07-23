import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.conversations.repositories import ConversationRepository
from app.features.conversations.types import Part
from lib.guid import GUIDManager
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration


def _repository(session: AsyncSession) -> ConversationRepository:
    return ConversationRepository(session, GUIDManager(), ClockManager())


async def test_get_or_create_persists_and_is_idempotent(session: AsyncSession) -> None:
    repository = _repository(session)

    first = await repository.get_or_create('telegram', '42', 'scout')
    again = await repository.get_or_create('telegram', '42', 'scout')

    assert first.id == again.id
    assert first.channel == 'telegram'
    assert first.id.version == 7
    assert first.created_at is not None


async def test_different_agent_creates_a_distinct_conversation(
    session: AsyncSession,
) -> None:
    repository = _repository(session)

    scout = await repository.get_or_create('telegram', '42', 'scout')
    sage = await repository.get_or_create('telegram', '42', 'sage')

    assert scout.id != sage.id


async def test_append_turn_round_trips_segment_kinds_and_jsonb(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', '99', 'scout')

    turn = await repository.append_turn(
        conversation,
        [
            Part(role='assistant', kind='text', payload={'text': 'checking'}),
            Part(role='assistant', kind='tool_call', payload={'name': 'search'}),
        ],
        usage={'input_tokens': 12, 'output_tokens': 3},
    )

    assert [(part.role, part.kind, dict(part.payload)) for part in turn.parts] == [
        ('assistant', 'text', {'text': 'checking'}),
        ('assistant', 'tool_call', {'name': 'search'}),
    ]
    assert turn.usage == {'input_tokens': 12, 'output_tokens': 3}


async def test_load_recent_turns_orders_by_time_and_limits(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', '7', 'scout')

    for index in range(3):
        await repository.append_turn(
            conversation,
            [Part(role='user', kind='text', payload={'text': str(index)})],
        )

    turns = await repository.load_recent_turns(conversation, limit=2)

    payloads = [dict(turn.parts[0].payload) for turn in turns]
    assert payloads == [{'text': '1'}, {'text': '2'}]


async def test_turns_are_scoped_to_their_conversation(session: AsyncSession) -> None:
    repository = _repository(session)
    scout = await repository.get_or_create('telegram', '1', 'scout')
    sage = await repository.get_or_create('telegram', '2', 'sage')

    await repository.append_turn(scout, [Part(role='user', kind='text')])

    assert len(await repository.load_recent_turns(scout, limit=10)) == 1
    assert await repository.load_recent_turns(sage, limit=10) == ()
