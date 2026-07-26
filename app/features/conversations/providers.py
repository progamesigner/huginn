from typing import override

from app.core.providers.database import DatabaseProvider
from app.core.providers.guid import GUIDProvider
from app.core.providers.llm import LLMProvider
from app.core.providers.temporal import ClockProvider
from lib.container import ContainerContract, ProviderContract

from .compaction import CompactionService
from .repositories import ConversationRepository


class ConversationProvider(ProviderContract):
    dependencies = (DatabaseProvider, GUIDProvider, LLMProvider, ClockProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            ConversationRepository,
            constructor=ConversationRepository,
            scoped=True,
        )
        container.bind(CompactionService, constructor=CompactionService, scoped=True)
