from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import override

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .contracts import DatabaseContract
from .exceptions import DatabaseError
from .session import ScopedSession


class DatabaseManager(DatabaseContract):
    engines: dict[str | None, AsyncEngine]
    session_makers: dict[str | None, async_sessionmaker[AsyncSession]]
    urls: dict[str | None, str]

    def __init__(self, urls: Mapping[str | None, str]) -> None:
        if not urls or any(name == '' or not url for name, url in urls.items()):
            raise DatabaseError('database urls are required')

        self.engines = {}
        self.session_makers = {}
        self.urls = dict(urls)

    @override
    def scoped(self, instance: str | None = None) -> ScopedSession:
        # NOTE: the caller owns the session lifetime
        return ScopedSession(self._async_session(instance)())

    @asynccontextmanager
    @override
    async def session(self, instance: str | None = None) -> AsyncIterator[AsyncSession]:
        async with self._async_session(instance)() as session:
            yield session

    async def dispose(self) -> None:
        for engine in self.engines.values():
            await engine.dispose()
        self.engines.clear()
        self.session_makers.clear()

    def url(self, instance: str | None = None) -> str:
        if instance not in self.urls:
            raise DatabaseError(f'database instance {instance!r} is not configured')

        return self.urls[instance]

    def _async_session(self, instance: str | None) -> async_sessionmaker[AsyncSession]:
        if instance not in self.session_makers:
            engine = create_async_engine(self.url(instance))
            self.engines[instance] = engine
            self.session_makers[instance] = async_sessionmaker(
                engine,
                expire_on_commit=False,
            )

        return self.session_makers[instance]
