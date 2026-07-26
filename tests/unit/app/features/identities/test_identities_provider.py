import pytest

from app.core.application import App
from app.core.providers.settings import SettingsProvider
from app.features.identities.exceptions import IdentityLinkSecretError
from app.features.identities.providers import IdentityProvider
from app.features.identities.repositories import IdentityRepository
from app.features.identities.services import IdentityLinkingService, IdentityResolver
from lib.container import ResolutionError
from tests.helpers.settings import make_settings


async def test_repository_resolves_scoped() -> None:
    settings = make_settings(
        database_url='postgresql+asyncpg://user:pass@localhost/huginn',
    )
    container = App().build(SettingsProvider(settings), IdentityProvider())

    with pytest.raises(ResolutionError, match='scoped'):
        container.resolve(IdentityRepository)

    async with container.scope() as scope:
        repository = scope.resolve(IdentityRepository)
        resolver = scope.resolve(IdentityResolver)
        assert resolver.repository is repository
        assert scope.resolve(IdentityRepository) is repository

    await container.dispose()


async def test_linking_service_requires_configured_secret() -> None:
    settings = make_settings(
        database_url='postgresql+asyncpg://user:pass@localhost/huginn',
    )
    container = App().build(SettingsProvider(settings), IdentityProvider())

    async with container.scope() as scope:
        with pytest.raises(IdentityLinkSecretError, match='not configured'):
            scope.resolve(IdentityLinkingService)

    await container.dispose()


async def test_linking_service_resolves_scoped_without_exposing_secret() -> None:
    settings = make_settings(
        database_url='postgresql+asyncpg://user:pass@localhost/huginn',
        identity_link_secret='x' * 32,
    )
    container = App().build(SettingsProvider(settings), IdentityProvider())

    async with container.scope() as scope:
        linking = scope.resolve(IdentityLinkingService)
        assert scope.resolve(IdentityLinkingService) is linking

    await container.dispose()
