import pytest

from app.core.application import App
from app.core.providers.settings import SettingsProvider
from app.features.identities.providers import IdentitiesProvider
from app.features.identities.repositories import IdentityRepository
from lib.container import ResolutionError
from tests.helpers.settings import make_settings


async def test_repository_resolves_scoped() -> None:
    settings = make_settings(
        database_url='postgresql+asyncpg://user:pass@localhost/huginn'
    )
    container = App().build(SettingsProvider(settings), IdentitiesProvider())

    with pytest.raises(ResolutionError, match='scoped'):
        container.resolve(IdentityRepository)

    async with container.scope() as scope:
        repository = scope.resolve(IdentityRepository)
        assert scope.resolve(IdentityRepository) is repository

    await container.dispose()
