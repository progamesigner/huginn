import asyncio
from datetime import timedelta

import pytest

from lib.redis import RedisError, RedisLock
from lib.redis.scripts import EXTEND, RELEASE
from tests.helpers.redis import FakeClient, make_redis

_TTL = timedelta(milliseconds=60)


def test_the_lua_resources_guard_on_the_owner_token() -> None:
    for source in (EXTEND, RELEASE):
        assert "redis.call('get', KEYS[1]) == ARGV[1]" in source

    assert "redis.call('pexpire', KEYS[1], ARGV[2])" in EXTEND
    assert "redis.call('del', KEYS[1])" in RELEASE


def test_rejects_a_non_positive_ttl() -> None:
    redis = make_redis(FakeClient())

    with pytest.raises(RedisError, match='lock ttl must be positive'):
        RedisLock(redis, 'chat:7', ttl=timedelta(0))


def test_owner_tokens_are_unique_per_lock() -> None:
    redis = make_redis(FakeClient())

    first = RedisLock(redis, 'chat:7', ttl=_TTL)
    second = RedisLock(redis, 'chat:7', ttl=_TTL)

    assert first.token != second.token


async def test_acquire_writes_the_owner_token_with_nx_and_the_lease() -> None:
    client = FakeClient()
    lock = RedisLock(make_redis(client), 'chat:7', ttl=_TTL)

    assert await lock.acquire() is True
    assert client.writes == [('chat:7', lock.token, True, 60)]


async def test_release_evaluates_the_script_against_the_owner_token() -> None:
    client = FakeClient()
    lock = RedisLock(make_redis(client), 'chat:7', ttl=_TTL)

    assert await lock.release() is True
    assert client.evaluated == [(RELEASE, ('chat:7',), (lock.token,))]


async def test_extend_hands_the_script_a_fresh_lease() -> None:
    client = FakeClient()
    lock = RedisLock(make_redis(client), 'chat:7', ttl=_TTL)

    assert await lock.extend() is True
    assert client.evaluated == [(EXTEND, ('chat:7',), (lock.token, 60))]


async def test_a_falsy_reply_means_the_lock_is_no_longer_ours() -> None:
    client = FakeClient(outcomes=[0, 0])
    lock = RedisLock(make_redis(client), 'chat:7', ttl=_TTL)

    assert await lock.extend() is False
    assert await lock.release() is False


async def test_watchdog_renews_the_lease_while_the_lock_is_held() -> None:
    client = FakeClient()
    lock = RedisLock(make_redis(client), 'chat:7', ttl=_TTL)

    async with lock.renew():
        await asyncio.sleep(_TTL.total_seconds() * 1.5)

    # NOTE: renewals run at ttl/3, so a 1.5x hold must have re-leased more than once
    assert len(client.evaluated) >= 2
    assert all(script == EXTEND for script, _, _ in client.evaluated)


async def test_watchdog_stops_once_the_lease_is_lost() -> None:
    client = FakeClient(outcomes=[0])
    lock = RedisLock(make_redis(client), 'chat:7', ttl=_TTL)

    async with lock.renew():
        await asyncio.sleep(_TTL.total_seconds() * 1.5)

    assert len(client.evaluated) == 1


async def test_operations_target_the_configured_alias() -> None:
    redis = make_redis(FakeClient())
    lock = RedisLock(redis, 'chat:7', ttl=_TTL, alias='queue')

    await lock.acquire()

    redis.connect.assert_called_once_with('queue')
