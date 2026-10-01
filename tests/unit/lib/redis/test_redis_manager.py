import pytest
from redis.asyncio import Redis

from lib.redis import RedisError, RedisManager


async def test_manager_builds_and_disposes_client() -> None:
    manager = RedisManager({None: 'redis://cache/0'})
    assert manager.clients == {}

    async with manager.connect() as connected:
        assert isinstance(connected, Redis)
        assert connected is manager.clients[None]

    await manager.dispose()
    assert manager.clients == {}


async def test_named_alias_uses_an_independent_client() -> None:
    manager = RedisManager(
        {
            None: 'redis://cache/0',
            'queue': 'redis://queue/1',
        },
    )

    async with manager.connect() as default, manager.connect('queue') as queue:
        assert default is manager.clients[None]
        assert queue is manager.clients['queue']
        assert default is not queue

    await manager.dispose()
    assert manager.clients == {}


async def test_unknown_alias_raises_redis_error() -> None:
    manager = RedisManager({None: 'redis://cache/0'})

    with pytest.raises(RedisError, match='queue'):
        async with manager.connect('queue'):
            pass


def test_empty_urls_are_rejected() -> None:
    with pytest.raises(RedisError, match='redis urls are required'):
        RedisManager({})
