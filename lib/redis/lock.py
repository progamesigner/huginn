import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from .contracts import RedisContract
from .exceptions import RedisError

logger = logging.getLogger(__name__)


def _lua(name: str) -> str:
    resources = Path(__file__).parent / 'resources'
    return (resources / f'{name}.lua').read_text(encoding='utf-8')


_EXTEND_LUA = _lua('extend')
_RELEASE_LUA = _lua('release')
_RENEWALS_PER_LEASE = 3


class RedisLock:
    instance: str | None
    key: str
    redis: RedisContract
    token: bytes
    ttl: timedelta

    def __init__(
        self,
        redis: RedisContract,
        key: str,
        *,
        ttl: timedelta,
        instance: str | None = None,
    ) -> None:
        if ttl <= timedelta(0):
            raise RedisError('lock ttl must be positive')

        self.instance = instance
        self.key = key
        self.redis = redis
        self.token = secrets.token_bytes(16)
        self.ttl = ttl

    async def acquire(self) -> bool:
        async with self.redis.connect(self.instance) as client:
            held = await client.set(self.key, self.token, nx=True, px=self._lease())
            return bool(held)

    async def extend(self) -> bool:
        async with self.redis.connect(self.instance) as client:
            script = client.register_script(_EXTEND_LUA)
            extended = await script(keys=[self.key], args=[self.token, self._lease()])
            return bool(extended)

    @asynccontextmanager
    async def renew(self) -> AsyncIterator[None]:
        watcher = asyncio.create_task(self._watch())
        try:
            yield
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    async def release(self) -> bool:
        async with self.redis.connect(self.instance) as client:
            script = client.register_script(_RELEASE_LUA)
            released = await script(keys=[self.key], args=[self.token])
            return bool(released)

    def _lease(self) -> int:
        return max(round(self.ttl / timedelta(milliseconds=1)), 1)

    async def _watch(self) -> None:
        interval = self.ttl.total_seconds() / _RENEWALS_PER_LEASE
        while True:
            await asyncio.sleep(interval)
            if not await self.extend():
                logger.warning('Lost redis lock %s before its lease renewed', self.key)
                return
