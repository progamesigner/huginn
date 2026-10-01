import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.application import App
from app.core.providers.database import DatabaseProvider
from lib.database import DatabaseContract, DatabaseError, DatabaseManager
from tests.helpers.settings import app_with, make_settings

_URL = 'postgresql+asyncpg://user:pass@localhost/huginn'


def _container() -> App:
    settings = make_settings(database_url=_URL)
    return app_with(settings).build(DatabaseProvider())


def test_binds_database_contract_as_shared_manager() -> None:
    container = _container()

    manager = container.resolve(DatabaseContract)
    assert isinstance(manager, DatabaseManager)
    assert container.resolve(DatabaseContract) is manager


def test_provider_defers_missing_url_until_resolution() -> None:
    container = app_with(make_settings()).build(
        DatabaseProvider(),
    )

    with pytest.raises(DatabaseError, match='not configured'):
        container.resolve(DatabaseContract)


def test_sessions_come_from_the_contract_not_the_container() -> None:
    container = _container()

    assert not container.bound(AsyncSession)
