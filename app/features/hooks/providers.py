from typing import override

from app.core.config.registry import HuginnConfig
from app.core.providers.http import HTTPClientProvider
from app.core.providers.temporal import ClockProvider
from app.features.agents.providers import AgentProvider
from app.features.mcp.providers import MCPProvider
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract
from lib.mcp import MCPClientContract
from lib.observability import Logger
from lib.temporal import ClockContract

from .config import HOOKS
from .sources.configured import ConfiguredHookSource


class HooksProvider(ProviderContract):
    dependencies = (AgentProvider, MCPProvider, HTTPClientProvider, ClockProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            ConfiguredHookSource,
            factory=lambda c: ConfiguredHookSource(
                c.resolve(HuginnConfig).section(HOOKS).root,
                c.resolve(MCPClientContract),
                c.resolve(HTTPClientContract),
                c.resolve(ClockContract),
                logger=c.resolve(Logger),
            ),
            shared=True,
        )
