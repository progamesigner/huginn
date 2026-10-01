from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import override

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .contracts import DatabaseContract
from .exceptions import DatabaseError


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

    @asynccontextmanager
    @override
    async def session(self, alias: str | None = None) -> AsyncIterator[AsyncSession]:
        async with self._async_session(alias)() as session:
            yield session

    @asynccontextmanager
    @override
    async def transaction(
        self,
        alias: str | None = None,
    ) -> AsyncIterator[AsyncSession]:
        async with self.session(alias) as session, session.begin():
            yield session

    @override
    async def ping(self, alias: str | None = None) -> None:
        async with self.session(alias) as session:
            await session.execute(text('SELECT 1'))

    async def dispose(self) -> None:
        for engine in self.engines.values():
            await engine.dispose()
        self.engines.clear()
        self.session_makers.clear()

    def _url(self, alias: str | None = None) -> str:
        if alias not in self.urls:
            raise DatabaseError(f'database alias {alias!r} is not configured')

        return self.urls[alias]

    def _async_session(self, alias: str | None) -> async_sessionmaker[AsyncSession]:
        if alias not in self.session_makers:
            engine = create_async_engine(self._url(alias))
            self.engines[alias] = engine
            self.session_makers[alias] = async_sessionmaker(
                engine,
                expire_on_commit=False,
            )

        return self.session_makers[alias]
