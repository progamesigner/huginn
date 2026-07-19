from typing import override

from app.core.providers.database import DatabaseProvider
from app.core.providers.guid import GUIDProvider
from lib.container import ContainerContract, ProviderContract

from .repositories import ConversationRepository


class ConversationsProvider(ProviderContract):
    dependencies = (DatabaseProvider, GUIDProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            ConversationRepository,
            constructor=ConversationRepository,
            scoped=True,
        )
