import pytest
from sqlalchemy.exc import IntegrityError
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


@pytest.mark.parametrize('kind', ['audio', 'image'], ids=['audio', 'image'])
async def test_a_media_part_is_within_the_declared_vocabulary(
    session: AsyncSession,
    kind: str,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', '77', 'scout')

    turn = await repository.append_turn(
        conversation,
        [Part(role='user', kind=kind, payload={'text': 'a placeholder'})],
    )

    assert [part.kind for part in turn.parts] == [kind]


@pytest.mark.parametrize(
    'part',
    [
        Part(role='narrator', kind='text'),
        Part(role='user', kind='interpretive-dance'),
    ],
    ids=['role', 'kind'],
)
async def test_a_part_outside_the_declared_vocabulary_is_refused(
    session: AsyncSession,
    part: Part,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', '13', 'scout')

    # NOTE: the reader branches on role and kind, so the database is where the vocabulary
    # has to be closed -- a writer that invents one must not be able to persist it
    with pytest.raises(IntegrityError):
        await repository.append_turn(conversation, [part])


async def test_turns_are_scoped_to_their_conversation(session: AsyncSession) -> None:
    repository = _repository(session)
    scout = await repository.get_or_create('telegram', '1', 'scout')
    sage = await repository.get_or_create('telegram', '2', 'sage')

    await repository.append_turn(scout, [Part(role='user', kind='text')])

    assert len(await repository.load_recent_turns(scout, limit=10)) == 1
    assert await repository.load_recent_turns(sage, limit=10) == ()


async def test_list_by_chat_returns_every_agent_of_that_chat(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    scout = await repository.get_or_create('telegram', '31', 'scout')
    sage = await repository.get_or_create('telegram', '31', 'sage')
    await repository.get_or_create('telegram', '32', 'scout')

    listed = await repository.list_by_chat('telegram', '31')

    # NOTE: a chat holds one conversation per agent, so a chat-level command has to see
    # all of them rather than guess which one the router last picked
    assert [conversation.id for conversation in listed] == [sage.id, scout.id]


async def test_reset_moves_the_boundary_to_the_newest_turn(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', '33', 'scout')
    for index in range(2):
        await repository.append_turn(
            conversation,
            [Part(role='user', kind='text', payload={'text': str(index)})],
        )
    newest = (await repository.load_recent_turns(conversation, limit=1))[0]

    reset = await repository.reset(conversation, conversation.compact_version)

    assert reset is not None
    assert reset.summarized_through_turn_id == newest.id
    assert reset.compact_version == conversation.compact_version + 1
    assert await repository.load_turns_since_summary(reset) == ()


async def test_reset_on_an_empty_conversation_holds_no_boundary(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', '34', 'scout')

    reset = await repository.reset(conversation, conversation.compact_version)

    assert reset is not None
    assert reset.summarized_through_turn_id is None


async def test_reset_loses_a_cas_race_without_moving_anything(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', '35', 'scout')
    await repository.append_turn(
        conversation,
        [Part(role='user', kind='text', payload={'text': 'hello'})],
    )

    # NOTE: a compaction running against the same chat bumps the version, and the reset
    # must not then cut history at a boundary that race decided
    assert (
        await repository.reset(conversation, conversation.compact_version + 1) is None
    )
