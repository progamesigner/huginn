from typing import override

from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.core.providers.database import DatabaseProvider
from app.core.providers.guid import GUIDProvider
from app.core.providers.llm import LLMProvider
from app.core.providers.temporal import ClockProvider
from app.features.agents.contracts import TranscriptStoreContract
from app.features.agents.providers import AgentProvider
from app.features.blobs.contracts import BlobStoreFactoryContract
from app.features.blobs.providers import BlobProvider
from lib.container import ContainerContract, ProviderContract
from lib.database import DatabaseContract
from lib.guid import GUIDContract
from lib.llm import LLMContract
from lib.temporal import ClockContract

from .contracts import ConversationControlContract
from .services import ConversationControl, ConversationTranscriptStore


def _control(container: ContainerContract) -> ConversationControl:
    return ConversationControl(
        container.resolve(DatabaseContract),
        container.resolve(GUIDContract),
        container.resolve(ClockContract),
        container.resolve(LLMContract),
        container.resolve(HuginnConfig),
    )


def _store(container: ContainerContract) -> ConversationTranscriptStore:
    return ConversationTranscriptStore(
        container.resolve(DatabaseContract),
        container.resolve(GUIDContract),
        container.resolve(ClockContract),
        container.resolve(LLMContract),
        container.resolve(HuginnConfig),
        # NOTE: the transcript owns the turn's session, blobs not reusing it
        container.resolve(BlobStoreFactoryContract),
    )


class ConversationProvider(ProviderContract):
    dependencies = (
        AgentProvider,
        BlobProvider,
        ClockProvider,
        ConfigProvider,
        DatabaseProvider,
        GUIDProvider,
        LLMProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            ConversationControlContract,
            factory=_control,
            shared=True,
        )
        container.bind(TranscriptStoreContract, factory=_store, shared=True)
