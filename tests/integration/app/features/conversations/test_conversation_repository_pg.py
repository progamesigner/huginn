from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.conversations.entities import Part
from app.features.conversations.repositories import ConversationRepository
from lib.guid import GUIDManager
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration


def _repository(session: AsyncSession) -> ConversationRepository:
    return ConversationRepository(session, GUIDManager(), ClockManager())


async def test_get_or_create_persists_and_is_idempotent(session: AsyncSession) -> None:
    repository = _repository(session)

    first = await repository.get_or_create('telegram', '42')
    again = await repository.get_or_create('telegram', '42')

    assert first.id == again.id
    assert first.channel == 'telegram'
    assert first.id.version == 7
    assert first.created_at is not None


async def test_same_chat_reuses_one_conversation(
    session: AsyncSession,
) -> None:
    repository = _repository(session)

    scout = await repository.get_or_create('telegram', '42')
    sage = await repository.get_or_create('telegram', '42')

    assert scout.id == sage.id


async def test_append_turn_round_trips_segment_kinds_and_jsonb(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', '99')

    turn = await repository.append_turn(
        conversation,
        'scout',
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
    assert turn.agent == 'scout'


async def test_load_recent_turns_orders_by_time_and_limits(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', '7')

    for index in range(3):
        await repository.append_turn(
            conversation,
            'scout',
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
    conversation = await repository.get_or_create('telegram', '77')

    turn = await repository.append_turn(
        conversation,
        'scout',
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
    conversation = await repository.get_or_create('telegram', '13')

    # NOTE: the reader branches on role and kind, so the database is where the vocabulary
    # has to be closed -- a writer that invents one must not be able to persist it
    with pytest.raises(IntegrityError):
        await repository.append_turn(conversation, 'scout', [part])


async def test_turns_are_scoped_to_their_conversation(session: AsyncSession) -> None:
    repository = _repository(session)
    scout = await repository.get_or_create('telegram', '1')
    sage = await repository.get_or_create('telegram', '2')

    await repository.append_turn(scout, 'scout', [Part(role='user', kind='text')])

    assert len(await repository.load_recent_turns(scout, limit=10)) == 1
    assert await repository.load_recent_turns(sage, limit=10) == ()


async def test_find_by_chat_returns_the_single_conversation(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    scout = await repository.get_or_create('telegram', '31')
    sage = await repository.get_or_create('telegram', '31')
    await repository.get_or_create('telegram', '32')

    found = await repository.find_by_chat('telegram', '31')

    assert scout.id == sage.id
    assert found is not None
    assert found.id == scout.id
    assert await repository.find_by_chat('telegram', 'missing') is None


async def test_latest_agent_reads_the_newest_turn(session: AsyncSession) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', 'latest')

    assert await repository.latest_agent(conversation) is None
    await repository.append_turn(
        conversation,
        'scout',
        [Part(role='user', kind='text')],
    )
    await repository.append_turn(
        conversation,
        'sage',
        [Part(role='assistant', kind='text')],
    )

    assert await repository.latest_agent(conversation) == 'sage'


async def test_reset_moves_the_boundary_to_the_newest_turn(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', '33')
    for index in range(2):
        await repository.append_turn(
            conversation,
            'scout',
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
    conversation = await repository.get_or_create('telegram', '34')

    reset = await repository.reset(conversation, conversation.compact_version)

    assert reset is not None
    assert reset.summarized_through_turn_id is None


async def test_reset_loses_a_cas_race_without_moving_anything(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', '35')
    await repository.append_turn(
        conversation,
        'scout',
        [Part(role='user', kind='text', payload={'text': 'hello'})],
    )

    # NOTE: a compaction running against the same chat bumps the version, and the reset
    # must not then cut history at a boundary that race decided
    assert (
        await repository.reset(conversation, conversation.compact_version + 1) is None
    )


async def test_document_blob_ids_span_the_whole_conversation(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', '36')
    other = await repository.get_or_create('telegram', '37')
    first = str(uuid4())
    second = str(uuid4())
    await repository.append_turn(
        conversation,
        'scout',
        [Part(role='user', kind='document', payload={'extracted_blob': first})],
    )
    folded = await repository.load_turns_since_summary(conversation)
    compacted = await repository.apply_compaction(
        conversation,
        summary_payload={'summary': 'a document was shared'},
        turns_folded=folded,
        expected_version=conversation.compact_version,
    )
    assert compacted is not None
    await repository.append_turn(
        compacted,
        'scout',
        [
            Part(role='user', kind='document', payload={'extracted_blob': second}),
            Part(role='user', kind='document', payload={'extracted_blob': None}),
            Part(role='user', kind='text', payload={'extracted_blob': 123}),
        ],
    )
    await repository.append_turn(
        other,
        'scout',
        [Part(role='user', kind='document', payload={'extracted_blob': str(uuid4())})],
    )

    assert await repository.load_document_blob_ids(compacted) == {first, second}


async def test_routing_samples_never_include_target_or_future_turns(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('routing-export', '1')
    turns = []
    for index in range(3):
        turns.append(
            await repository.append_turn(
                conversation,
                'scout',
                [Part('user', 'text', {'text': str(index)})],
            ),
        )
    await repository.append_turn(
        conversation,
        'scout',
        [Part('assistant', 'summary', {'summary': 'future summary'})],
    )
    other = await repository.get_or_create('different', '1')
    await repository.append_turn(
        other,
        'scout',
        [Part('user', 'text', {'text': 'not in channel'})],
    )
    samples = [
        sample
        async for sample in repository.routing_samples(
            'routing-export',
            datetime(2000, 1, 1, tzinfo=UTC),
            2,
        )
    ]
    assert [turn.id for turn, _ in samples] == [turn.id for turn in turns]
    assert [tuple(turn.id for turn in prior) for _, prior in samples] == [
        (),
        (turns[0].id,),
        (turns[0].id, turns[1].id),
    ]
    recent = await repository.load_routing_turns(conversation, 2)
    assert [turn.id for turn in recent] == [turns[1].id, turns[2].id]


async def test_a_reset_hides_the_documents_shared_before_it(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('telegram', '38')
    before = str(uuid4())
    after = str(uuid4())
    await repository.append_turn(
        conversation,
        'scout',
        [Part(role='user', kind='document', payload={'extracted_blob': before})],
    )

    reset = await repository.reset(conversation, conversation.compact_version)
    assert reset is not None
    assert await repository.load_document_blob_ids(reset) == frozenset()

    await repository.append_turn(
        reset,
        'scout',
        [Part(role='user', kind='document', payload={'extracted_blob': after})],
    )
    assert await repository.load_document_blob_ids(reset) == {after}
