from collections.abc import Sequence
from hashlib import sha256
from typing import override
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config.registry import HuginnConfig
from app.features.agents.types import PromptAttachment, TurnPrompt
from app.features.conversations.config import CompactionConfig
from app.features.conversations.contracts import BlobStoreContract
from app.features.conversations.repositories import (
    BlobRepository,
    ConversationRepository,
)
from app.features.conversations.schemas.sqlalchemy import BlobRecord
from app.features.conversations.services import (
    ConversationTranscript,
    ConversationTranscriptStore,
)
from app.features.conversations.types import Blob, Conversation, Part
from lib.database import DatabaseContract, DatabaseManager
from lib.guid import GUIDManager
from lib.llm import (
    ImageSegment,
    LLMClient,
    LLMContract,
    Message,
    ProviderConfig,
    TextSegment,
)
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration

_PHOTO = b'\xff\xd8\xff\xe0 not really a jpeg'
_VOICE = b'OggS not really an ogg'


class NoModels(LLMContract):
    @override
    def client_for(self, model: str) -> LLMClient | None:
        return None

    @override
    def provider_for(self, model: str) -> ProviderConfig | None:
        return None


class FlakyBlobs(BlobStoreContract):
    blobs: BlobRepository
    stored: int

    def __init__(self, session: AsyncSession) -> None:
        self.blobs = BlobRepository(session, GUIDManager())
        self.stored = 0

    @override
    async def store(self, mime: str, data: bytes) -> Blob:
        self.stored += 1
        if self.stored > 1:
            raise RuntimeError('blob store is down')

        return await self.blobs.store(mime, data)


async def _blob(session: AsyncSession, id_: str) -> bytes:
    query = select(BlobRecord).where(BlobRecord.id == UUID(id_))

    return (await session.execute(query)).scalar_one().data


def _blobs(session: AsyncSession) -> BlobRepository:
    return BlobRepository(session, GUIDManager())


def _repository(session: AsyncSession) -> ConversationRepository:
    return ConversationRepository(session, GUIDManager(), ClockManager())


def _store(database: DatabaseContract) -> ConversationTranscriptStore:
    return ConversationTranscriptStore(
        database,
        GUIDManager(),
        ClockManager(),
        NoModels(),
        HuginnConfig(sections={'compaction': CompactionConfig()}),
        _blobs,
    )


def _transcript(
    session: AsyncSession,
    conversation: Conversation,
) -> ConversationTranscript:
    return ConversationTranscript(_repository(session), _blobs(session), conversation)


def _prompt(text: str) -> TurnPrompt:
    return TurnPrompt(text=text, actor='user:1')


def _photo(
    hint: str = '[image: image/jpeg 8x8]',
    data: bytes = _PHOTO,
) -> PromptAttachment:
    return PromptAttachment(
        kind='image',
        mime='image/jpeg',
        reference='file-1',
        unique='unique-1',
        data=data,
        hint=hint,
    )


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
    transcript = _transcript(session, conversation)

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
    transcript = _transcript(session, conversation)

    await transcript.record([_prompt('hello')], '')

    assert _texts(await transcript.history()) == [('user', 'hello')]


async def test_history_opens_with_the_summary_and_omits_folded_turns(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '2', 'scout')
    transcript = _transcript(session, conversation)
    await transcript.record([_prompt('ancient')], 'forgotten')
    folded = await repository.load_turns_since_summary(conversation)
    updated = await repository.apply_compaction(
        conversation,
        summary_payload={'summary': 'they discussed the ancients'},
        turns_folded=folded,
        expected_version=conversation.compact_version,
    )

    assert updated is not None
    after = _transcript(session, updated)
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

    assert _texts(await _transcript(session, conversation).history()) == [
        ('user', 'search it'),
        ('assistant', 'here you go'),
    ]


async def test_an_attached_photo_is_stored_as_a_blob_and_a_media_part(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '5', 'scout')

    await _transcript(session, conversation).record(
        [TurnPrompt(text='what is this?', actor='user:1', attachments=(_photo(),))],
        'a duck',
    )

    turn = (await repository.load_recent_turns(conversation, limit=1))[0]
    text, media, answer = turn.parts
    assert (text.role, text.kind) == ('user', 'text')
    assert (media.role, media.kind) == ('user', 'image')
    assert (answer.role, answer.kind) == ('assistant', 'text')

    payload = dict(media.payload)
    stored = payload.pop('blob')
    # NOTE: the payload carries the reference and the digest, never the bytes -- that is
    # what keeps compaction and the token estimate off a base64 blob
    assert payload == {
        'text': '[image: image/jpeg 8x8]',
        'actor': 'user:1',
        'sha256': sha256(_PHOTO).hexdigest(),
        'mime': 'image/jpeg',
        'size': len(_PHOTO),
        'reference': 'file-1',
        'unique': 'unique-1',
    }
    assert await _blob(session, str(stored)) == _PHOTO


async def test_a_stored_photo_replays_as_its_placeholder(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '6', 'scout')
    transcript = _transcript(session, conversation)
    await transcript.record(
        [TurnPrompt(text='', actor='user:1', attachments=(_photo(),))],
        'a duck',
    )

    history = await transcript.history()

    # NOTE: the bytes are kept but never replayed -- a later round costs the placeholder,
    # not a second upload of the same image
    assert _texts(history) == [
        ('user', '[image: image/jpeg 8x8]'),
        ('assistant', 'a duck'),
    ]
    assert not any(
        isinstance(segment, ImageSegment)
        for message in history
        for segment in message.segments
    )


async def test_a_transcribed_voice_note_is_stored_as_what_it_said(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '7', 'scout')
    transcript = _transcript(session, conversation)
    voice = PromptAttachment(
        kind='audio',
        mime='audio/ogg',
        reference='file-2',
        unique='unique-2',
        data=_VOICE,
        hint='[audio: audio/ogg 12s]',
        transcript='remind me at six',
    )

    await transcript.record(
        [TurnPrompt(text='', actor='user:1', attachments=(voice,))],
        'ok',
    )

    turn = (await repository.load_recent_turns(conversation, limit=1))[0]
    # NOTE: the words are what a later round replays, and the ogg stays in its blob
    assert turn.parts[0].kind == 'audio'
    assert turn.parts[0].payload['text'] == 'remind me at six'
    assert _texts(await transcript.history()) == [
        ('user', 'remind me at six'),
        ('assistant', 'ok'),
    ]


@pytest.mark.usefixtures('_migrated')
async def test_recording_a_round_commits_it_on_its_own(database_url: str) -> None:
    database = DatabaseManager({None: database_url})
    store = _store(database)
    try:
        await store.record('support_bot', '7777', 'scout', [_prompt('hello')], 'answer')

        # NOTE: a separate call is a separate session, so seeing the turn back proves the
        # record committed rather than merely flushed
        assert _texts(await store.history('support_bot', '7777', 'scout')) == [
            ('user', 'hello'),
            ('assistant', 'answer'),
        ]
    finally:
        await database.dispose()


# NOTE: the store used to hold one transaction across the whole turn, so a turn that
# failed after answering rolled its own round away and the next turn had no memory of it
@pytest.mark.usefixtures('_migrated')
async def test_a_recorded_round_survives_a_later_failure(database_url: str) -> None:
    database = DatabaseManager({None: database_url})
    store = _store(database)
    try:
        await store.record(
            'support_bot',
            '8888',
            'scout',
            [_prompt('hello')],
            'partial',
        )

        with pytest.raises(RuntimeError):
            raise RuntimeError('the turn died after answering')

        assert _texts(await store.history('support_bot', '8888', 'scout')) == [
            ('user', 'hello'),
            ('assistant', 'partial'),
        ]
    finally:
        await database.dispose()


# NOTE: splitting the round's transaction in two must not make either half leaky -- the
# first photo is stored before the second one fails, and has to disappear with it
@pytest.mark.usefixtures('_migrated')
async def test_a_record_that_raises_partway_commits_nothing(database_url: str) -> None:
    database = DatabaseManager({None: database_url})
    store = ConversationTranscriptStore(
        database,
        GUIDManager(),
        ClockManager(),
        NoModels(),
        HuginnConfig(sections={'compaction': CompactionConfig()}),
        FlakyBlobs,
    )
    prompt = TurnPrompt(
        text='look',
        actor='user:1',
        attachments=(_photo(), _photo(data=b'\xff\xd8\xff\xe0 another jpeg')),
    )
    try:
        with pytest.raises(RuntimeError, match='blob store'):
            await store.record('support_bot', '9999', 'scout', [prompt], 'answer')

        async with database.session() as session:
            assert (await session.execute(select(BlobRecord))).scalars().all() == []

        assert await store.history('support_bot', '9999', 'scout') == ()
    finally:
        await database.dispose()
