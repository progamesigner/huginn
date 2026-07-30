from typing import override

from app.core.providers.database import DatabaseProvider
from app.core.providers.guid import GUIDProvider
from app.core.providers.llm import LLMProvider
from app.core.providers.temporal import ClockProvider
from app.features.agents.contracts import TranscriptStoreContract
from app.features.agents.providers import AgentProvider
from lib.container import ContainerContract, ProviderContract

from .compaction import CompactionService
from .contracts import BlobStoreContract
from .repositories import BlobRepository, ConversationRepository
from .services import ConversationTranscriptStore


class ConversationProvider(ProviderContract):
    dependencies = (
        AgentProvider,
        ClockProvider,
        DatabaseProvider,
        GUIDProvider,
        LLMProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            ConversationRepository,
            constructor=ConversationRepository,
            scoped=True,
        )
        container.bind(BlobStoreContract, constructor=BlobRepository, scoped=True)
        container.bind(CompactionService, constructor=CompactionService, scoped=True)
        container.bind(
            TranscriptStoreContract,
            constructor=ConversationTranscriptStore,
            shared=True,
        )
