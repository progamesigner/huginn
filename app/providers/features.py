from typing import override

from app.features.agents.providers import AgentsConfigProvider
from app.features.conversations.providers import ConversationsProvider
from app.features.identities.providers import IdentitiesProvider
from app.features.tools.providers import ToolProvider
from lib.container import ContainerContract, ProviderContract


class FeaturesProvider(ProviderContract):
    dependencies = (
        AgentsConfigProvider,
        ConversationsProvider,
        IdentitiesProvider,
        ToolProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
