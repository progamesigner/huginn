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
    async def connect(self, instance: str | None = None) -> AsyncIterator[Redis[bytes]]:
        if instance not in self.clients:
            self.clients[instance] = Redis.from_url(self.url(instance))

        yield self.clients[instance]

    async def dispose(self) -> None:
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()

    def url(self, instance: str | None = None) -> str:
        if instance not in self.urls:
            raise RedisError(f'redis instance {instance!r} is not configured')

        return self.urls[instance]
