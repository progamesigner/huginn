from typing import override

from app.features.agents.providers import AgentsConfigProvider
from app.features.channels.providers import ChannelProvider
from app.features.conversations.providers import ConversationsProvider
from app.features.identities.providers import IdentitiesProvider
from app.features.skills.providers import SkillsConfigProvider
from app.features.tools.providers import ToolProvider
from lib.container import ContainerContract, ProviderContract


class FeaturesProvider(ProviderContract):
    dependencies = (
        AgentsConfigProvider,
        ConversationsProvider,
        IdentitiesProvider,
        SkillsConfigProvider,
        ToolProvider,
        ChannelProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
