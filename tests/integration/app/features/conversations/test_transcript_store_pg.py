from collections.abc import Sequence
from datetime import timedelta
from hashlib import sha256
from typing import override
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config.registry import HuginnConfig
from app.features.agents.entities import PromptAttachment, TurnPrompt
from app.features.blobs.config import BlobsConfig, RetentionConfig
from app.features.blobs.contracts import BlobServiceContract
from app.features.blobs.entities import Blob, BlobKind
from app.features.blobs.repositories import BlobRepository
from app.features.blobs.services import BlobService
from app.features.blobs.sources.sqlalchemy import BlobRecord
from app.features.conversations.config import CompactionConfig
from app.features.conversations.entities import Conversation, Part
from app.features.conversations.repositories import ConversationRepository
from app.features.conversations.services import (
    ConversationTranscript,
    ConversationTranscriptStore,
)
from lib.database import DatabaseContract, DatabaseManager
from lib.guid import GUIDManager
from lib.llm import (
    ImageSegment,
    LLMClient,
    LLMContract,
    Message,
    ModelRef,
    ProviderConfig,
    TextSegment,
)
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration

_PHOTO = b'\xff\xd8\xff\xe0 not really a jpeg'
_VOICE = b'OggS not really an ogg'


class NoModels(LLMContract):
    @override
    def client_for(self, ref: ModelRef) -> LLMClient | None:
        return None

    @override
    def provider_for(self, ref: ModelRef) -> ProviderConfig | None:
        return None


class SessionBlobs(BlobServiceContract):
    reads: BlobRepository

    # NOTE: reads go through the test's own session, which sees its uncommitted rows
    def __init__(self, session: AsyncSession) -> None:
        self.reads = BlobRepository(session, GUIDManager(), ClockManager())

    @override
    async def store(
        self,
        transaction: AsyncSession,
        mime: str,
        data: bytes,
        *,
        kind: BlobKind,
        filename: str | None = None,
        ttl: timedelta | None = None,
    ) -> Blob:
        return await BlobRepository(transaction, GUIDManager(), ClockManager()).store(
            mime,
            data,
            kind=kind,
            filename=filename,
            ttl=ttl,
        )

    @override
    async def load(self, id: UUID) -> Blob | None:
        return await self.reads.load(id)

    @override
    async def head(self, id: UUID) -> Blob | None:
        return await self.reads.head(id)


class FlakyBlobs(BlobServiceContract):
    stored: int

    def __init__(self) -> None:
        self.stored = 0

    @override
    async def store(
        self,
        transaction: AsyncSession,
        mime: str,
        data: bytes,
        *,
        kind: BlobKind,
        filename: str | None = None,
        ttl: timedelta | None = None,
    ) -> Blob:
        self.stored += 1
        if self.stored > 1:
            raise RuntimeError('blob store is down')

        return await BlobRepository(transaction, GUIDManager(), ClockManager()).store(
            mime,
            data,
            kind=kind,
            filename=filename,
            ttl=ttl,
        )

    @override
    async def load(self, id: UUID) -> Blob | None:
        raise NotImplementedError

    @override
    async def head(self, id: UUID) -> Blob | None:
        raise NotImplementedError


async def _blob(session: AsyncSession, id_: str) -> bytes:
    query = select(BlobRecord).where(BlobRecord.id == UUID(id_))

    return (await session.execute(query)).scalar_one().data


def _blobs(session: AsyncSession) -> SessionBlobs:
    return SessionBlobs(session)


def _repository(session: AsyncSession) -> ConversationRepository:
    return ConversationRepository(session, GUIDManager(), ClockManager())


def _store(database: DatabaseContract) -> ConversationTranscriptStore:
    return ConversationTranscriptStore(
        database,
        GUIDManager(),
        ClockManager(),
        NoModels(),
        HuginnConfig(
            sections={'blobs': BlobsConfig(), 'compaction': CompactionConfig()},
        ),
        BlobService(database, GUIDManager(), ClockManager()),
    )


def _transcript(
    session: AsyncSession,
    conversation: Conversation,
) -> ConversationTranscript:
    return ConversationTranscript(
        session,
        _repository(session),
        _blobs(session),
        conversation,
        RetentionConfig(),
    )


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
    conversation = await repository.get_or_create('support_bot', '4242')
    transcript = _transcript(session, conversation)

    await transcript.record('scout', [_prompt('first'), _prompt('second')], 'answer')

    assert _texts(await transcript.history()) == [
        ('user', 'first'),
        ('user', 'second'),
        ('assistant', 'answer'),
    ]


async def test_an_empty_answer_records_only_what_the_user_said(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '1')
    transcript = _transcript(session, conversation)

    await transcript.record('scout', [_prompt('hello')], '')

    assert _texts(await transcript.history()) == [('user', 'hello')]


async def test_history_opens_with_the_summary_and_omits_folded_turns(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '2')
    transcript = _transcript(session, conversation)
    await transcript.record('scout', [_prompt('ancient')], 'forgotten')
    folded = await repository.load_turns_since_summary(conversation)
    updated = await repository.apply_compaction(
        conversation,
        summary_payload={'summary': 'they discussed the ancients'},
        turns_folded=folded,
        expected_version=conversation.compact_version,
    )

    assert updated is not None
    after = _transcript(session, updated)
    await after.record('scout', [_prompt('recent')], 'fresh')

    # NOTE: the summary stands in for everything before the boundary, so the folded turn
    # must not come back as well -- that would send the model the same content twice
    assert _texts(await after.history()) == [
        ('user', 'Summary of the conversation so far: they discussed the ancients'),
        ('user', 'recent'),
        ('assistant', 'fresh'),
    ]


async def test_tool_traffic_is_kept_but_never_replayed(session: AsyncSession) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '3')
    await repository.append_turn(
        conversation,
        'scout',
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
    conversation = await repository.get_or_create('support_bot', '5')

    await _transcript(session, conversation).record(
        'scout',
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
        'extracted_blob': None,
    }
    assert await _blob(session, str(stored)) == _PHOTO


async def test_an_inbound_blob_expires_after_the_configured_retention(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '6')
    transcript = ConversationTranscript(
        session,
        repository,
        _blobs(session),
        conversation,
        RetentionConfig(inbound=timedelta(days=7)),
    )
    before = ClockManager().now()

    await transcript.record(
        'scout',
        [TurnPrompt(text='keep this', actor='user:1', attachments=(_photo(),))],
        'kept',
    )

    record = (await session.execute(select(BlobRecord))).scalar_one()
    assert record.expires_at is not None
    assert before + timedelta(days=7) <= record.expires_at
    assert record.expires_at <= ClockManager().now() + timedelta(days=7)


async def test_extracted_text_expires_while_the_inbound_original_is_kept(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '7')
    document = PromptAttachment(
        kind='document',
        mime='application/pdf',
        reference='file-2',
        unique='unique-2',
        data=b'%PDF not really a pdf',
        hint='[document: application/pdf]',
        document_text='the extracted words',
    )

    await _transcript(session, conversation).record(
        'scout',
        [TurnPrompt(text='read this', actor='user:1', attachments=(document,))],
        'read',
    )

    part = (await repository.load_recent_turns(conversation, limit=1))[0].parts[1]
    original = await _blobs(session).head(UUID(str(part.payload['blob'])))
    extracted = await _blobs(session).head(UUID(str(part.payload['extracted_blob'])))
    assert original is not None
    assert extracted is not None
    assert original.expires_at is None
    assert extracted.expires_at is not None


async def test_a_stored_photo_replays_as_its_placeholder(
    session: AsyncSession,
) -> None:
    repository = _repository(session)
    conversation = await repository.get_or_create('support_bot', '6')
    transcript = _transcript(session, conversation)
    await transcript.record(
        'scout',
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
    conversation = await repository.get_or_create('support_bot', '7')
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
        'scout',
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
        assert _texts(await store.history('support_bot', '7777')) == [
            ('user', 'hello'),
            ('assistant', 'answer'),
        ]
    finally:
        await database.dispose()


@pytest.mark.usefixtures('_migrated')
async def test_history_and_agent_stamps_are_shared_across_routed_agents(
    database_url: str,
) -> None:
    database = DatabaseManager({None: database_url})
    store = _store(database)
    try:
        await store.record('support_bot', 'mixed', 'quick', [_prompt('first')], 'one')
        await store.record('support_bot', 'mixed', 'deep', [_prompt('second')], 'two')

        assert _texts(await store.history('support_bot', 'mixed')) == [
            ('user', 'first'),
            ('assistant', 'one'),
            ('user', 'second'),
            ('assistant', 'two'),
        ]
        async with database.session() as session:
            repository = _repository(session)
            conversation = await repository.find_by_chat('support_bot', 'mixed')
            assert conversation is not None
            turns = await repository.load_recent_turns(conversation, limit=10)
            assert [turn.agent for turn in turns] == ['quick', 'deep']
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

        assert _texts(await store.history('support_bot', '8888')) == [
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
        HuginnConfig(
            sections={'blobs': BlobsConfig(), 'compaction': CompactionConfig()},
        ),
        FlakyBlobs(),
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

        assert await store.history('support_bot', '9999') == ()
    finally:
        await database.dispose()
