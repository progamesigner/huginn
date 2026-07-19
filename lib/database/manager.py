from collections.abc import AsyncIterator
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
    engine: AsyncEngine | None
    sessions: async_sessionmaker[AsyncSession] | None
    url: str

    def __init__(self, url: str) -> None:
        self.engine = None
        self.sessions = None
        self.url = url

    def scoped(self) -> ScopedSession:
        # NOTE: the caller owns the session lifetime
        return ScopedSession(self._factory()())

    @asynccontextmanager
    @override
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._factory()() as session:
            yield session

    async def dispose(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()
        self.engine = None
        self.sessions = None

    def _factory(self) -> async_sessionmaker[AsyncSession]:
        if not self.url:
            raise DatabaseError('database url is not configured')

        if self.sessions is None:
            self.engine = create_async_engine(self.url)
            self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

        return self.sessions
