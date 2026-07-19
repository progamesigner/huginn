import pytest

from app.core.application import App
from app.features.conversations.providers import ConversationsProvider
from app.features.conversations.repositories import ConversationRepository
from lib.container import ResolutionError

_URL = 'postgresql+asyncpg://user:pass@localhost/huginn'


async def test_repository_resolves_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('HUGINN_DATABASE_URL', _URL)
    container = App().build(ConversationsProvider())

    with pytest.raises(ResolutionError, match='scoped'):
        container.resolve(ConversationRepository)

    async with container.scope() as scope:
        repository = scope.resolve(ConversationRepository)
        assert isinstance(repository, ConversationRepository)
        assert scope.resolve(ConversationRepository) is repository

    await container.dispose()
