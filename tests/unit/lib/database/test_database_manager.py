from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from lib.database import DatabaseError, DatabaseManager, ScopedSession

_URL = 'postgresql+asyncpg://user:pass@localhost/huginn'


async def test_session_context_manager_yields_a_session() -> None:
    manager = DatabaseManager({None: _URL})

    async with manager.session() as session:
        assert isinstance(session, AsyncSession)

    await manager.dispose()


async def test_scoped_returns_a_disposable_session_holder() -> None:
    manager = DatabaseManager({None: _URL})

    scoped = manager.scoped()
    assert isinstance(scoped.session, AsyncSession)

    await scoped.dispose()
    await manager.dispose()


async def test_manager_reuses_engine_and_sessionmaker() -> None:
    manager = DatabaseManager({None: _URL})

    manager.scoped()
    engine = manager.engines[None]
    session_maker = manager.session_makers[None]
    manager.scoped()

    assert manager.engines[None] is engine
    assert manager.session_makers[None] is session_maker
    await manager.dispose()


async def test_named_instance_uses_an_independent_engine() -> None:
    manager = DatabaseManager(
        {
            None: _URL,
            'analytics': _URL.replace('huginn', 'analytics'),
        },
    )

    manager.scoped()
    manager.scoped('analytics')

    assert manager.engines[None] is not manager.engines['analytics']
    await manager.dispose()
    assert manager.engines == {}
    assert manager.session_makers == {}


def test_unknown_instance_raises_database_error() -> None:
    manager = DatabaseManager({None: _URL})

    with pytest.raises(DatabaseError, match='analytics'):
        manager.scoped('analytics')


def test_empty_urls_raise_database_error() -> None:
    with pytest.raises(DatabaseError, match='database urls are required'):
        DatabaseManager({})


async def test_dispose_without_engine_is_a_noop() -> None:
    await DatabaseManager({None: _URL}).dispose()


async def test_scoped_session_dispose_closes_the_session() -> None:
    closed = []

    class FakeSession:
        async def close(self) -> None:
            closed.append(True)

    holder = ScopedSession(cast(AsyncSession, FakeSession()))
    await holder.dispose()

    assert closed == [True]
