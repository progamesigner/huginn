from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from lib.database import DatabaseError, DatabaseManager, ScopedSession

_URL = 'postgresql+asyncpg://user:pass@localhost/huginn'


async def test_session_context_manager_yields_a_session() -> None:
    manager = DatabaseManager(_URL)

    async with manager.session() as session:
        assert isinstance(session, AsyncSession)

    await manager.dispose()


async def test_scoped_returns_a_disposable_session_holder() -> None:
    manager = DatabaseManager(_URL)

    scoped = manager.scoped()
    assert isinstance(scoped.session, AsyncSession)

    await scoped.dispose()
    await manager.dispose()


async def test_manager_reuses_engine_and_sessionmaker() -> None:
    manager = DatabaseManager(_URL)

    manager.scoped()
    engine = manager.engine
    manager.scoped()

    assert manager.engine is engine
    await manager.dispose()


def test_missing_url_raises_database_error() -> None:
    manager = DatabaseManager('')

    with pytest.raises(DatabaseError):
        manager.scoped()


async def test_dispose_without_engine_is_a_noop() -> None:
    await DatabaseManager('').dispose()


async def test_scoped_session_dispose_closes_the_session() -> None:
    closed = []

    class FakeSession:
        async def close(self) -> None:
            closed.append(True)

    holder = ScopedSession(cast(AsyncSession, FakeSession()))
    await holder.dispose()

    assert closed == [True]
