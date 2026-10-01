from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import override

from redis.asyncio import Redis

from .contracts import RedisContract
from .exceptions import RedisError


class RedisManager(RedisContract):
    clients: dict[str | None, Redis[bytes]]
    urls: dict[str | None, str]

    def __init__(self, urls: Mapping[str | None, str]) -> None:
        if not urls or any(name == '' or not url for name, url in urls.items()):
            raise RedisError('redis urls are required')

        self.clients = {}
        self.urls = dict(urls)

    @asynccontextmanager
    @override
    async def connect(self, alias: str | None = None) -> AsyncIterator[Redis[bytes]]:
        if alias not in self.clients:
            self.clients[alias] = Redis.from_url(self.url(alias))

        yield self.clients[alias]

    async def dispose(self) -> None:
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()

    def url(self, alias: str | None = None) -> str:
        if alias not in self.urls:
            raise RedisError(f'redis alias {alias!r} is not configured')

        return self.urls[alias]
