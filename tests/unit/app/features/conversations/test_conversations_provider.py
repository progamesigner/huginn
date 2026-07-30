import pytest

from app.core.application import App
from app.core.providers.settings import SettingsProvider
from app.features.conversations.contracts import BlobStoreContract
from app.features.conversations.providers import ConversationProvider
from app.features.conversations.repositories import (
    BlobRepository,
    ConversationRepository,
)
from lib.container import ResolutionError
from tests.helpers.settings import make_settings

_URL = 'postgresql+asyncpg://user:pass@localhost/huginn'


async def test_repository_resolves_scoped() -> None:
    settings = make_settings(database_url=_URL)
    container = App().build(SettingsProvider(settings), ConversationProvider())

    with pytest.raises(ResolutionError, match='scoped'):
        container.resolve(ConversationRepository)

    async with container.scope() as scope:
        repository = scope.resolve(ConversationRepository)
        assert isinstance(repository, ConversationRepository)
        assert scope.resolve(ConversationRepository) is repository

    await container.dispose()


async def test_the_blob_store_resolves_scoped_to_the_pg_backend() -> None:
    settings = make_settings(database_url=_URL)
    container = App().build(SettingsProvider(settings), ConversationProvider())

    with pytest.raises(ResolutionError, match='scoped'):
        container.resolve(BlobStoreContract)

    async with container.scope() as scope:
        # NOTE: the contract is the seam another backend swaps in at, so what the
        # binding resolves to is the thing worth asserting
        store = scope.resolve(BlobStoreContract)
        assert isinstance(store, BlobRepository)
        assert scope.resolve(BlobStoreContract) is store

    await container.dispose()
