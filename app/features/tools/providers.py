from typing import override

from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.providers.config import ConfigProvider
from app.core.providers.http import HTTPClientProvider
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract
from lib.mcp import (
    Connector,
    MCPClientContract,
    MCPManager,
    MCPServer,
    streamable_http_connector,
)

from .config import build_sections
from .mcp.config import MCP_SERVERS, MCPServerConfig, MCPServersConfig
from .registry import ToolRegistry
from .websearch.config import WebSearchToolConfig
from .websearch.repositories import WebSearchRepository
from .websearch.tool import WebSearchTool, build_websearch


def _headers(server: MCPServerConfig) -> dict[str, str]:
    if server.auth is None:
        return {}

    return {'Authorization': f'Bearer {server.auth.get_secret_value()}'}


def build_servers(config: MCPServersConfig) -> list[MCPServer]:
    return [
        MCPServer(name=name, url=str(server.url), headers=_headers(server))
        for name, server in config.root.items()
    ]


class MCPConfigProvider(ProviderContract):
    dependencies = (ConfigProvider,)

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

    def _build(self, container: ContainerContract) -> MCPManager:
        servers = build_servers(container.resolve(HuginnConfig).section(MCP_SERVERS))
        connector = self.connector or streamable_http_connector(
            container.resolve(HTTPClientContract)
        )
        return MCPManager(servers, connector)


class ToolRegistryProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ToolRegistry, instance=ToolRegistry())


class ToolsConfigProvider(ProviderContract):
    dependencies = (ConfigProvider, ToolRegistryProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        sections = container.resolve(SectionRegistry)
        for spec in build_sections(container.resolve(ToolRegistry)):
            sections.register(spec)


class ToolProvider(ProviderContract):
    dependencies = (HTTPClientProvider, ToolRegistryProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        registry = container.resolve(ToolRegistry)
        http = container.resolve(HTTPClientContract)

        repository = WebSearchRepository(http)
        container.bind(WebSearchRepository, instance=repository)
        registry.register(
            WebSearchTool,
            WebSearchToolConfig,
            builder=lambda config: build_websearch(repository, config),
        )
