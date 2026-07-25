import asyncio
from collections.abc import Sequence
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from lib.redis import RedisContract, RedisError, RedisLock
from lib.redis.lock import _EXTEND_LUA, _RELEASE_LUA

_TTL = timedelta(milliseconds=60)


class FakeScript:
    client: FakeClient
    script: str

    def __init__(self, client: FakeClient, script: str) -> None:
        self.client = client
        self.script = script

    async def __call__(
        self,
        keys: Sequence[str] | None = None,
        args: Sequence[object] | None = None,
    ) -> int:
        self.client.evaluated.append(
            (self.script, tuple(keys or ()), tuple(args or ()))
        )
        return self.client.outcomes.pop(0) if self.client.outcomes else 1


class FakeClient:
    evaluated: list[tuple[str, tuple[str, ...], tuple[object, ...]]]
    outcomes: list[int]
    writes: list[tuple[str, bytes, bool, int | None]]

    def __init__(self, outcomes: Sequence[int] = ()) -> None:
        self.evaluated = []
        self.outcomes = list(outcomes)
        self.writes = []

    async def set(
        self,
        key: str,
        value: bytes,
        *,
        nx: bool = False,
        px: int | None = None,
    ) -> bool | None:
        self.writes.append((key, value, nx, px))
        return True

    def register_script(self, script: str) -> FakeScript:
        return FakeScript(self, script)


def make_redis(client: FakeClient) -> MagicMock:
    connection = MagicMock()
    connection.__aenter__ = AsyncMock(return_value=client)
    connection.__aexit__ = AsyncMock(return_value=False)

    redis = MagicMock(spec=RedisContract)
    redis.connect.return_value = connection
    return redis


def test_the_lua_resources_guard_on_the_owner_token() -> None:
    for source in (_EXTEND_LUA, _RELEASE_LUA):
        assert "redis.call('get', KEYS[1]) == ARGV[1]" in source

    assert "redis.call('pexpire', KEYS[1], ARGV[2])" in _EXTEND_LUA
    assert "redis.call('del', KEYS[1])" in _RELEASE_LUA


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
    assert client.evaluated == [(_RELEASE_LUA, ('chat:7',), (lock.token,))]


async def test_extend_hands_the_script_a_fresh_lease() -> None:
    client = FakeClient()
    lock = RedisLock(make_redis(client), 'chat:7', ttl=_TTL)

    assert await lock.extend() is True
    assert client.evaluated == [(_EXTEND_LUA, ('chat:7',), (lock.token, 60))]


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
    assert all(script == _EXTEND_LUA for script, _, _ in client.evaluated)


async def test_watchdog_stops_once_the_lease_is_lost() -> None:
    client = FakeClient(outcomes=[0])
    lock = RedisLock(make_redis(client), 'chat:7', ttl=_TTL)

    async with lock.renew():
        await asyncio.sleep(_TTL.total_seconds() * 1.5)

    assert len(client.evaluated) == 1


async def test_operations_target_the_configured_instance() -> None:
    redis = make_redis(FakeClient())
    lock = RedisLock(redis, 'chat:7', ttl=_TTL, instance='queue')

    await lock.acquire()

    redis.connect.assert_called_once_with('queue')
