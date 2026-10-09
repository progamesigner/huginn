from datetime import timedelta
from hashlib import sha256
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.blobs.exceptions import BlobError
from app.features.blobs.repositories import BlobRepository
from app.features.blobs.sources.sqlalchemy import BlobGrantRecord, BlobRecord
from app.features.conversations.entities import Part
from app.features.conversations.repositories import (
    ConversationRepository,
)
from app.features.conversations.services import ConversationBlobReferences
from app.features.conversations.sources.sqlalchemy import (
    ConversationRecord,
)
from lib.guid import GUIDManager
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration

_PHOTO = b'\xff\xd8\xff\xe0 not really a jpeg'
_VOICE = b'OggS not really an ogg'


def _blobs(session: AsyncSession) -> BlobRepository:
    return BlobRepository(session, GUIDManager(), ClockManager())


def _conversations(session: AsyncSession) -> ConversationRepository:
    return ConversationRepository(session, GUIDManager(), ClockManager())


async def _count(session: AsyncSession) -> int:
    query = select(func.count()).select_from(BlobRecord)

    return (await session.execute(query)).scalar_one()


async def _record(session: AsyncSession, digest: str) -> BlobRecord:
    query = select(BlobRecord).where(BlobRecord.sha256 == digest)

    return (await session.execute(query)).scalar_one()


async def test_store_keeps_the_bytes_and_what_they_are(session: AsyncSession) -> None:
    blob = await _blobs(session).store(
        'image/jpeg',
        _PHOTO,
        kind='image',
        filename='photo.jpg',
    )

    assert blob.sha256 == sha256(_PHOTO).hexdigest()
    assert blob.mime == 'image/jpeg'
    assert blob.size == len(_PHOTO)
    assert blob.id.version == 7
    assert blob.kind == 'image'
    assert blob.filename == 'photo.jpg'
    assert (await _record(session, blob.sha256)).data == _PHOTO


async def test_load_returns_data_but_head_does_not_fetch_it(
    session: AsyncSession,
) -> None:
    repository = _blobs(session)
    blob = await repository.store('image/jpeg', _PHOTO, kind='image')

    loaded = await repository.load(blob.id)
    headed = await repository.head(blob.id)

    assert loaded is not None
    assert loaded.data == _PHOTO
    assert headed is not None
    assert headed.data is None
    assert await repository.load(UUID(int=99)) is None


async def test_duplicate_storage_keeps_the_longest_retention(
    session: AsyncSession,
) -> None:
    repository = _blobs(session)
    first = await repository.store(
        'image/jpeg',
        _PHOTO,
        kind='image',
        ttl=timedelta(days=1),
    )
    permanent = await repository.store('image/jpeg', _PHOTO, kind='image')

    assert first.id == permanent.id
    assert permanent.expires_at is None


async def test_the_same_bytes_stored_twice_stay_one_row(session: AsyncSession) -> None:
    repository = _blobs(session)

    first = await repository.store('image/jpeg', _PHOTO, kind='image')
    again = await repository.store('image/jpeg', _PHOTO, kind='image')

    # NOTE: forwarding the same photo again must cost nothing; the digest is what
    # identifies the bytes, so the second store has to land on the first row
    assert first == again
    assert await _count(session) == 1


async def test_different_bytes_get_their_own_row(session: AsyncSession) -> None:
    repository = _blobs(session)

    photo = await repository.store('image/jpeg', _PHOTO, kind='image')
    voice = await repository.store('audio/ogg', _VOICE, kind='audio')

    assert photo.id != voice.id
    assert photo.sha256 != voice.sha256
    assert await _count(session) == 2


async def test_a_deleted_conversation_leaves_its_blobs_behind(
    session: AsyncSession,
) -> None:
    conversation = await _conversations(session).get_or_create('telegram', '5')
    blob = await _blobs(session).store('image/jpeg', _PHOTO, kind='image')
    await _conversations(session).append_turn(
        conversation,
        'scout',
        [Part(role='user', kind='image', payload={'blob': str(blob.id)})],
    )

    await session.execute(
        delete(ConversationRecord).where(ConversationRecord.id == conversation.id),
    )

    # NOTE: a part points at a blob through its payload, not a foreign key, so the
    # cascade stops at the turn -- there is no reclaim path yet and this pins that
    assert await _count(session) == 1


async def test_prune_keeps_expired_blobs_that_a_part_still_references(
    session: AsyncSession,
) -> None:
    blobs = _blobs(session)
    day = timedelta(days=1)
    original = await blobs.store('application/pdf', b'%PDF', kind='document', ttl=day)
    extracted = await blobs.store('text/markdown', b'text', kind='document', ttl=day)
    orphan = await blobs.store('image/jpeg', _PHOTO, kind='image', ttl=day)
    conversation = await _conversations(session).get_or_create('telegram', '6')
    await _conversations(session).append_turn(
        conversation,
        'scout',
        [
            Part(
                role='user',
                kind='document',
                payload={
                    'blob': str(original.id),
                    'extracted_blob': str(extracted.id),
                },
            ),
        ],
    )
    later = ClockManager().now() + 2 * day
    references = (ConversationBlobReferences(GUIDManager(), ClockManager()),)

    assert await blobs.prune(later, references, dry_run=True) == 1
    assert await blobs.prune(later, references) == 1

    remaining = set((await session.execute(select(BlobRecord.id))).scalars())
    assert remaining == {original.id, extracted.id}
    assert orphan.id not in remaining


async def test_prune_refuses_to_run_without_a_reference_source(
    session: AsyncSession,
) -> None:
    blobs = _blobs(session)
    await blobs.store('image/jpeg', _PHOTO, kind='image', ttl=timedelta(days=1))

    with pytest.raises(BlobError, match='refusing to prune'):
        await blobs.prune(ClockManager().now() + timedelta(days=2), ())

    assert await _count(session) == 1


async def test_a_grant_names_one_owner_and_repeats_harmlessly(
    session: AsyncSession,
) -> None:
    blobs = _blobs(session)
    blob = await blobs.store('image/jpeg', _PHOTO, kind='image')

    await blobs.grant(blob.id, 'web-key')
    await blobs.grant(blob.id, 'web-key')

    assert await blobs.granted(blob.id, 'web-key')
    assert not await blobs.granted(blob.id, 'other-key')
    count = select(func.count()).select_from(BlobGrantRecord)
    assert (await session.execute(count)).scalar_one() == 1


async def test_pruning_a_granted_blob_takes_its_grants_along(
    session: AsyncSession,
) -> None:
    blobs = _blobs(session)
    day = timedelta(days=1)
    blob = await blobs.store('image/jpeg', _PHOTO, kind='image', ttl=day)
    await blobs.grant(blob.id, 'web-key')
    references = (ConversationBlobReferences(GUIDManager(), ClockManager()),)

    assert await blobs.prune(ClockManager().now() + 2 * day, references) == 1

    assert not await blobs.granted(blob.id, 'web-key')
