import asyncio
import secrets
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

from lib.redis import RedisLock, RedisManager
from tests.helpers.logging import make_logger

pytestmark = pytest.mark.integration

_TTL = timedelta(milliseconds=300)


@pytest.fixture
async def redis(redis_url: str) -> AsyncIterator[RedisManager]:
    manager = RedisManager({None: redis_url})
    try:
        yield manager
    finally:
        await manager.dispose()


@pytest.fixture
def key() -> str:
    return f'huginn:test:lock:{secrets.token_hex(8)}'


async def test_only_one_contender_holds_the_lock(redis: RedisManager, key: str) -> None:
    owner = RedisLock(redis, key, ttl=_TTL, logger=make_logger())
    rival = RedisLock(redis, key, ttl=_TTL, logger=make_logger())

    assert await owner.acquire() is True
    assert await rival.acquire() is False

    assert await owner.release() is True
    assert await rival.acquire() is True
    assert await rival.release() is True


async def test_an_unrenewed_lease_expires_and_frees_the_lock(
    redis: RedisManager,
    key: str,
) -> None:
    abandoned = RedisLock(redis, key, ttl=_TTL, logger=make_logger())
    successor = RedisLock(redis, key, ttl=_TTL, logger=make_logger())

    assert await abandoned.acquire() is True
    await asyncio.sleep(_TTL.total_seconds() * 2)

    # NOTE: this is the crash-recovery path — a dead owner never releases
    assert await successor.acquire() is True
    assert await successor.release() is True


async def test_extend_keeps_the_lock_past_its_original_ttl(
    redis: RedisManager,
    key: str,
) -> None:
    owner = RedisLock(redis, key, ttl=_TTL, logger=make_logger())
    rival = RedisLock(redis, key, ttl=_TTL, logger=make_logger())

    assert await owner.acquire() is True
    await asyncio.sleep(_TTL.total_seconds() * 0.6)
    assert await owner.extend() is True
    await asyncio.sleep(_TTL.total_seconds() * 0.6)

    assert await rival.acquire() is False
    assert await owner.release() is True


async def test_a_non_owner_can_neither_extend_nor_release(
    redis: RedisManager,
    key: str,
) -> None:
    owner = RedisLock(redis, key, ttl=_TTL, logger=make_logger())
    rival = RedisLock(redis, key, ttl=_TTL, logger=make_logger())

    assert await owner.acquire() is True

    assert await rival.extend() is False
    assert await rival.release() is False

    assert await rival.acquire() is False
    assert await owner.release() is True


async def test_release_after_expiry_never_deletes_the_next_owners_lock(
    redis: RedisManager,
    key: str,
) -> None:
    lapsed = RedisLock(redis, key, ttl=_TTL, logger=make_logger())
    successor = RedisLock(redis, key, ttl=_TTL, logger=make_logger())

    assert await lapsed.acquire() is True
    await asyncio.sleep(_TTL.total_seconds() * 2)
    assert await successor.acquire() is True

    # NOTE: the whole point of the owner token — a late release must be a no-op
    assert await lapsed.release() is False
    assert await successor.release() is True


async def test_the_watchdog_holds_the_lock_through_a_long_turn(
    redis: RedisManager,
    key: str,
) -> None:
    owner = RedisLock(redis, key, ttl=_TTL, logger=make_logger())
    rival = RedisLock(redis, key, ttl=_TTL, logger=make_logger())

    assert await owner.acquire() is True

    async with owner.renew():
        await asyncio.sleep(_TTL.total_seconds() * 3)
        assert await rival.acquire() is False

    assert await owner.release() is True


async def test_the_lock_lapses_once_the_watchdog_stops(
    redis: RedisManager,
    key: str,
) -> None:
    owner = RedisLock(redis, key, ttl=_TTL, logger=make_logger())
    successor = RedisLock(redis, key, ttl=_TTL, logger=make_logger())

    assert await owner.acquire() is True
    async with owner.renew():
        await asyncio.sleep(_TTL.total_seconds())

    await asyncio.sleep(_TTL.total_seconds() * 2)

    assert await successor.acquire() is True
    assert await successor.release() is True
