from typing import override

from app.core.config.exceptions import MissingConfigFileError
from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.core.providers.http import HTTPClientProvider
from app.features.agents.providers import AgentProvider
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract
from lib.mcp import (
    MCPClientContract,
    MCPManager,
    MCPServer,
    MCPTransport,
    streamable_http_connector,
)
from lib.observability import Logger

from .config import MCP_SERVERS, MCPServerConfig, MCPServerConfigList


def _build_servers(config: MCPServerConfigList) -> list[MCPServer]:
    return [
        MCPServer(name=name, url=str(server.url), headers=_headers(server))
        for name, server in config.root.items()
    ]


def _headers(server: MCPServerConfig) -> dict[str, str]:
    if server.auth is None:
        return {}

    return {'Authorization': f'Bearer {server.auth.get_secret_value()}'}


def _transport(container: ContainerContract) -> MCPTransport:
    return MCPTransport(
        streamable_http_connector(container.resolve(HTTPClientContract)),
    )


class MCPProvider(ProviderContract):
    dependencies = (AgentProvider, ConfigProvider, HTTPClientProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        # NOTE: a transport bound before build wins, which is how tests dial in memory
        if not container.bound(MCPTransport):
            container.bind(MCPTransport, factory=_transport, shared=True)
        container.bind(
            MCPClientContract,
            factory=self._build,
            shared=True,
            eager=True,
        )

    def _build(self, container: ContainerContract) -> MCPManager:
        connector = container.resolve(MCPTransport).connector

        try:
            config = container.resolve(HuginnConfig)
        except MissingConfigFileError:
            # NOTE: no config file at all means no servers; a malformed one still fails
            return MCPManager(
                [],
                connector,
                logger=container.resolve(Logger).bind(component='mcp'),
            )

        return MCPManager(
            _build_servers(config.section(MCP_SERVERS)),
            connector,
            logger=container.resolve(Logger).bind(component='mcp'),
        )
