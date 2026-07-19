import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.application import App
from app.core.providers.database import DatabaseProvider
from lib.container import ResolutionError
from lib.database import DatabaseContract, DatabaseManager

_URL = 'postgresql+asyncpg://user:pass@localhost/huginn'


def _container(monkeypatch: pytest.MonkeyPatch) -> App:
    monkeypatch.setenv('HUGINN_DATABASE_URL', _URL)
    return App().build(DatabaseProvider())


def test_binds_database_contract_as_shared_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _container(monkeypatch)

    manager = container.resolve(DatabaseContract)
    assert isinstance(manager, DatabaseManager)
    assert container.resolve(DatabaseContract) is manager


def test_session_binding_requires_an_active_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _container(monkeypatch)

    with pytest.raises(ResolutionError, match='scoped'):
        container.resolve(AsyncSession)


async def test_scope_yields_a_single_session_then_disposes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _container(monkeypatch)

    async with container.scope() as scope:
        session = scope.resolve(AsyncSession)
        assert isinstance(session, AsyncSession)
        assert scope.resolve(AsyncSession) is session

    async with container.scope() as other:
        assert other.resolve(AsyncSession) is not session

    await container.dispose()
