import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from lib.database import DatabaseError, DatabaseManager

_URL = 'postgresql+asyncpg://user:pass@localhost/huginn'


async def test_session_context_manager_yields_a_session() -> None:
    manager = DatabaseManager({None: _URL})

    async with manager.session() as session:
        assert isinstance(session, AsyncSession)

    await manager.dispose()


async def test_transaction_yields_a_session_inside_a_transaction() -> None:
    manager = DatabaseManager({None: _URL})

    async with manager.transaction() as session:
        assert session.in_transaction()

    await manager.dispose()


async def test_manager_reuses_engine_and_sessionmaker() -> None:
    manager = DatabaseManager({None: _URL})

    async with manager.session():
        engine = manager.engines[None]
        session_maker = manager.session_makers[None]
    async with manager.session():
        pass

    assert manager.engines[None] is engine
    assert manager.session_makers[None] is session_maker
    await manager.dispose()


async def test_named_alias_uses_an_independent_engine() -> None:
    manager = DatabaseManager(
        {
            None: _URL,
            'analytics': _URL.replace('huginn', 'analytics'),
        },
    )

    async with manager.session(), manager.session('analytics'):
        pass

    assert manager.engines[None] is not manager.engines['analytics']
    await manager.dispose()
    assert manager.engines == {}
    assert manager.session_makers == {}


async def test_unknown_alias_raises_database_error() -> None:
    manager = DatabaseManager({None: _URL})

    with pytest.raises(DatabaseError, match='analytics'):
        async with manager.session('analytics'):
            pass


def test_empty_urls_raise_database_error() -> None:
    with pytest.raises(DatabaseError, match='database urls are required'):
        DatabaseManager({})


async def test_dispose_without_engine_is_a_noop() -> None:
    await DatabaseManager({None: _URL}).dispose()
