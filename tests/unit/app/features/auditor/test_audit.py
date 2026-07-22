from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast, override
from unittest.mock import MagicMock

from redis.asyncio import Redis

from app.features.auditor.services import RedisAuditor
from lib.redis import RedisContract


class FakeRedis(RedisContract):
    def __init__(self, keys: tuple[bytes, ...]) -> None:
        client = MagicMock()

        async def scan_iter(**kwargs: object) -> AsyncIterator[bytes]:
            assert kwargs == {'count': 1000}
            for key in keys:
                yield key

        client.scan_iter = scan_iter
        self.client = cast('Redis[bytes]', client)

    @asynccontextmanager
    @override
    async def connect(self) -> AsyncIterator[Redis[bytes]]:
        yield self.client


async def test_audit_classifies_every_key_and_lists_other() -> None:
    auditor = RedisAuditor(
        FakeRedis(
            (
                b'agent:memory:channel:telegram:user:42:messages',
                b'BifrostSemanticCachePlugin:abc',
                b'registration:\xff',
            )
        )
    )

    report = await auditor.run()

    assert report.total == 3
    assert report.legacy_sessions == 1
    assert report.bifrost_cache == 1
    assert report.other == (r'registration:\xff',)
