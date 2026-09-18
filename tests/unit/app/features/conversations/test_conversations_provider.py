from pathlib import Path

import pytest

from app.core.application import App
from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.core.providers.settings import SettingsProvider
from app.features.blobs.contracts import BlobReferenceRegistry, BlobStoreContract
from app.features.blobs.repositories import BlobRepository
from app.features.conversations.config import COMPACTION
from app.features.conversations.providers import ConversationProvider
from app.features.conversations.repositories import ConversationRepository
from lib.container import ResolutionError
from tests.helpers.settings import make_settings

_URL = 'postgresql+asyncpg://user:pass@localhost/huginn'

_COMPACTION_OVERLAY = """
defaults:
  model: gpt-5-mini
compaction:
  summary_prompt: Summarize the discussion in one terse sentence.
agents:
  scout:
    description: Research agent
    instructions: You are scout.
"""


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


async def test_conversations_answer_which_blobs_they_still_reference() -> None:
    settings = make_settings(database_url=_URL)
    container = App().build(SettingsProvider(settings), ConversationProvider())

    async with container.scope() as scope:
        session = scope.resolve(BlobRepository).session
        references = container.resolve(BlobReferenceRegistry).for_session(session)

        assert [type(reference) for reference in references] == [
            ConversationRepository,
        ]

    await container.dispose()


async def test_conversations_config_provider_threads_a_configured_summary_prompt(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'huginn.yaml'
    path.write_text(_COMPACTION_OVERLAY)
    settings = make_settings(database_url=_URL)
    container = App().build(
        SettingsProvider(settings),
        ConfigProvider(path),
        ConversationProvider(),
    )

    config = container.resolve(HuginnConfig)

    assert (
        config.section(COMPACTION).summary_prompt
        == 'Summarize the discussion in one terse sentence.'
    )

    await container.dispose()
