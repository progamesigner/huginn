from pathlib import Path

from app.core.config.registry import HuginnConfig
from app.features.blobs.contracts import BlobReferenceRegistry, BlobStoreContract
from app.features.blobs.services import BlobService
from app.features.conversations.config import COMPACTION
from app.features.conversations.providers import ConversationProvider
from app.features.conversations.repositories import ConversationRepository
from lib.database import DatabaseContract
from tests.helpers.settings import app_with, make_settings

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


async def test_repositories_are_built_per_transaction_not_bound() -> None:
    settings = make_settings(database_url=_URL)
    container = app_with(settings).build(ConversationProvider())

    assert not container.bound(ConversationRepository)

    await container.dispose()


async def test_the_blob_store_resolves_without_a_transaction() -> None:
    settings = make_settings(database_url=_URL)
    container = app_with(settings).build(ConversationProvider())

    # NOTE: a turn's tools resolve this from the root, so it opens its own session
    store = container.resolve(BlobStoreContract)
    assert isinstance(store, BlobService)
    assert container.resolve(BlobStoreContract) is store

    await container.dispose()


async def test_conversations_answer_which_blobs_they_still_reference() -> None:
    settings = make_settings(database_url=_URL)
    container = app_with(settings).build(ConversationProvider())

    async with container.resolve(DatabaseContract).session() as session:
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
    container = app_with(settings.model_copy(update={'config_file': path})).build(
        ConversationProvider(),
    )

    config = container.resolve(HuginnConfig)

    assert (
        config.section(COMPACTION).summary_prompt
        == 'Summarize the discussion in one terse sentence.'
    )

    await container.dispose()
