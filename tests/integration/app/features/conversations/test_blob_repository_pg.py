from hashlib import sha256

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.conversations.repositories import (
    BlobRepository,
    ConversationRepository,
)
from app.features.conversations.schemas.sqlalchemy import (
    BlobRecord,
    ConversationRecord,
)
from app.features.conversations.types import Part
from lib.guid import GUIDManager
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration

_PHOTO = b'\xff\xd8\xff\xe0 not really a jpeg'
_VOICE = b'OggS not really an ogg'


def _blobs(session: AsyncSession) -> BlobRepository:
    return BlobRepository(session, GUIDManager())


def _conversations(session: AsyncSession) -> ConversationRepository:
    return ConversationRepository(session, GUIDManager(), ClockManager())


async def _count(session: AsyncSession) -> int:
    query = select(func.count()).select_from(BlobRecord)

    return (await session.execute(query)).scalar_one()


async def _record(session: AsyncSession, digest: str) -> BlobRecord:
    query = select(BlobRecord).where(BlobRecord.sha256 == digest)

    return (await session.execute(query)).scalar_one()


async def test_store_keeps_the_bytes_and_what_they_are(session: AsyncSession) -> None:
    blob = await _blobs(session).store('image/jpeg', _PHOTO)

    assert blob.sha256 == sha256(_PHOTO).hexdigest()
    assert blob.mime == 'image/jpeg'
    assert blob.size == len(_PHOTO)
    assert blob.id.version == 7
    assert (await _record(session, blob.sha256)).data == _PHOTO


async def test_the_same_bytes_stored_twice_stay_one_row(session: AsyncSession) -> None:
    repository = _blobs(session)

    first = await repository.store('image/jpeg', _PHOTO)
    again = await repository.store('image/jpeg', _PHOTO)

    # NOTE: forwarding the same photo again must cost nothing; the digest is what
    # identifies the bytes, so the second store has to land on the first row
    assert first == again
    assert await _count(session) == 1


async def test_different_bytes_get_their_own_row(session: AsyncSession) -> None:
    repository = _blobs(session)

    photo = await repository.store('image/jpeg', _PHOTO)
    voice = await repository.store('audio/ogg', _VOICE)

    assert photo.id != voice.id
    assert photo.sha256 != voice.sha256
    assert await _count(session) == 2


async def test_a_deleted_conversation_leaves_its_blobs_behind(
    session: AsyncSession,
) -> None:
    conversation = await _conversations(session).get_or_create('telegram', '5')
    blob = await _blobs(session).store('image/jpeg', _PHOTO)
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
