from typing import override

from app.features.agents.providers import AgentProvider
from app.features.auditor.providers import AuditorProvider
from app.features.channels.providers import ChannelProvider
from app.features.channels.telegram.providers import TelegramProvider
from app.features.conversations.providers import ConversationProvider
from app.features.identities.providers import IdentityProvider
from app.features.mcp.providers import MCPProvider
from app.features.skills.providers import SkillsProvider
from app.features.tools.providers import ToolProvider
from lib.container import ContainerContract, ProviderContract


class HuginnProvider(ProviderContract):
    dependencies = (
        AgentProvider,
        AuditorProvider,
        ChannelProvider,
        ConversationProvider,
        IdentityProvider,
        MCPProvider,
        SkillsProvider,
        TelegramProvider,
        ToolProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
