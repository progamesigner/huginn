import pytest

from app.features.identities.contracts import IdentityLinkingServiceContract
from app.features.identities.exceptions import IdentityLinkSecretError
from app.features.identities.providers import IdentityProvider
from app.features.identities.repositories import IdentityRepository
from app.features.identities.tokens import IdentityTokenCodec
from tests.helpers.settings import app_with, make_settings


async def test_repositories_are_built_per_transaction_not_bound() -> None:
    settings = make_settings(
        database_url='postgresql+asyncpg://user:pass@localhost/huginn',
    )
    container = app_with(settings).build(IdentityProvider())

    assert not container.bound(IdentityRepository)

    await container.dispose()


async def test_the_linking_contract_resolves_without_a_scope() -> None:
    # NOTE: a telegram command runs in a background turn, and nothing opens a scope
    # there -- so `/register` and `/start link_…` are exactly what a scoped dependency
    # on this path breaks, which is how it reached staging unnoticed
    settings = make_settings(
        database_url='postgresql+asyncpg://user:pass@localhost/huginn',
        identity_link_secret='x' * 32,
    )
    container = app_with(settings).build(IdentityProvider())

    linker = container.resolve(IdentityLinkingServiceContract)

    assert container.resolve(IdentityLinkingServiceContract) is linker

    await container.dispose()


async def test_token_codec_requires_configured_secret() -> None:
    settings = make_settings(
        database_url='postgresql+asyncpg://user:pass@localhost/huginn',
    )
    container = app_with(settings).build(IdentityProvider())

    with pytest.raises(IdentityLinkSecretError, match='not configured'):
        container.resolve(IdentityTokenCodec)

    await container.dispose()


async def test_token_codec_is_shared_once_configured() -> None:
    settings = make_settings(
        database_url='postgresql+asyncpg://user:pass@localhost/huginn',
        identity_link_secret='x' * 32,
    )
    container = app_with(settings).build(IdentityProvider())

    assert container.resolve(IdentityTokenCodec) is container.resolve(
        IdentityTokenCodec,
    )

    await container.dispose()
