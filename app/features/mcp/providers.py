from typing import override

from app.core.config.exceptions import MissingConfigFileError
from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.providers.config import ConfigProvider
from app.core.providers.http import HTTPClientProvider
from app.features.agents.contracts import ToolSourceRegistry
from app.features.agents.providers import AgentProvider
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract
from lib.mcp import (
    Connector,
    MCPClientContract,
    MCPManager,
    MCPServer,
    streamable_http_connector,
)

from .config import MCP_SERVERS, MCPServerConfig, MCPServerConfigList
from .sources import MCPToolSource


def _build_servers(config: MCPServerConfigList) -> list[MCPServer]:
    return [
        MCPServer(name=name, url=str(server.url), headers=_headers(server))
        for name, server in config.root.items()
    ]


def _headers(server: MCPServerConfig) -> dict[str, str]:
    if server.auth is None:
        return {}

    return {'Authorization': f'Bearer {server.auth.get_secret_value()}'}


class MCPConfigProvider(ProviderContract):
    dependencies = (AgentProvider, ConfigProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.resolve(SectionRegistry).register(MCP_SERVERS)


class MCPProvider(ProviderContract):
    dependencies = (HTTPClientProvider, MCPConfigProvider)

    connector: Connector | None

    def __init__(self, connector: Connector | None = None) -> None:
        self.connector = connector

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            MCPClientContract,
            factory=self._build,
            shared=True,
            eager=True,
        )
        container.resolve(ToolSourceRegistry).register(
            lambda: MCPToolSource(container.resolve(MCPClientContract)),
        )

    def _build(self, container: ContainerContract) -> MCPManager:
        connector = self.connector or streamable_http_connector(
            container.resolve(HTTPClientContract),
        )

        try:
            config = container.resolve(HuginnConfig)
        except MissingConfigFileError:
            # NOTE: no config file at all means no servers; a malformed one still fails
            return MCPManager([], connector)

        return MCPManager(_build_servers(config.section(MCP_SERVERS)), connector)
