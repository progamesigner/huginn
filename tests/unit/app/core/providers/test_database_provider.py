import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.application import App
from app.core.providers.database import DatabaseProvider
from app.core.providers.settings import SettingsProvider
from lib.container import ResolutionError
from lib.database import DatabaseContract, DatabaseManager
from tests.helpers.settings import make_settings

_URL = 'postgresql+asyncpg://user:pass@localhost/huginn'


def _container() -> App:
    settings = make_settings(database_url=_URL)
    return App().build(SettingsProvider(settings), DatabaseProvider())


def test_binds_database_contract_as_shared_manager() -> None:
    container = _container()

    manager = container.resolve(DatabaseContract)
    assert isinstance(manager, DatabaseManager)
    assert container.resolve(DatabaseContract) is manager


def test_session_binding_requires_an_active_scope() -> None:
    container = _container()

    with pytest.raises(ResolutionError, match='scoped'):
        container.resolve(AsyncSession)


async def test_scope_yields_a_single_session_then_disposes_it() -> None:
    container = _container()

    async with container.scope() as scope:
        session = scope.resolve(AsyncSession)
        assert isinstance(session, AsyncSession)
        assert scope.resolve(AsyncSession) is session

    async with container.scope() as other:
        assert other.resolve(AsyncSession) is not session

    await container.dispose()
