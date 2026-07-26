from typing import override

from app.core.providers.config import ConfigCommandProvider
from app.features.agents.providers import AgentsConfigProvider
from app.features.auditor.providers import AuditCommandProvider, AuditorProvider
from app.features.channels.providers import ChannelEndpointProvider, ChannelProvider
from app.features.channels.telegram.providers import TelegramProvider
from app.features.conversations.providers import ConversationsProvider
from app.features.identities.providers import (
    IdentitiesCommandProvider,
    IdentitiesProvider,
)
from app.features.probes.providers import ProbesEndpointProvider
from app.features.skills.providers import SkillsProvider
from app.features.tools.providers import MCPProvider, ToolProvider
from lib.container import ContainerContract, ProviderContract


class HuginnProvider(ProviderContract):
    dependencies = (
        AgentsConfigProvider,
        AuditorProvider,
        ChannelProvider,
        ConversationsProvider,
        IdentitiesProvider,
        MCPProvider,
        SkillsProvider,
        TelegramProvider,
        ToolProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass


class HuginnEndpointProvider(ProviderContract):
    dependencies = (ChannelEndpointProvider, ProbesEndpointProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass


class HuginnCommandProvider(ProviderContract):
    dependencies = (
        AuditCommandProvider,
        ConfigCommandProvider,
        IdentitiesCommandProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
