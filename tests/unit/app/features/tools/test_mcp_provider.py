import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from app.core.application import App
from app.core.providers.config import ConfigProvider
from app.features.tools.mcp.config import MCPServersConfig
from app.features.tools.providers import MCPProvider, build_servers
from lib.mcp import MCPClientContract, MCPServer, UnknownMCPServerError

_CONFIG = """
mcp_servers:
  memory:
    transport: http
    url: https://memory.test
    auth: secret-token
"""


def _echo_server() -> FastMCP:
    server = FastMCP('test')

    @server.tool()
    def echo(text: str) -> str:
        """Echo the text back to the caller."""
        return f'echoed: {text}'

    return server


@asynccontextmanager
async def _memory_connector(_: MCPServer) -> AsyncIterator[ClientSession]:
    async with create_connected_server_and_client_session(_echo_server()) as session:
        yield session


def test_build_servers_maps_auth_to_bearer_header() -> None:
    config = MCPServersConfig.model_validate_json(
        json.dumps(
            {
                'memory': {
                    'transport': 'http',
                    'url': 'https://memory.test',
                    'auth': 'secret-token',
                },
                'stock': {'transport': 'http', 'url': 'https://stock.test'},
            },
        ),
    )

    servers = {server.name: server for server in build_servers(config)}

    assert servers['memory'].headers == {'Authorization': 'Bearer secret-token'}
    assert servers['stock'].headers == {}
    assert servers['memory'].url == 'https://memory.test/'


async def test_provider_connects_configured_servers(tmp_path: Path) -> None:
    config = tmp_path / 'huginn.yaml'
    config.write_text(_CONFIG)

    container = App().build(
        ConfigProvider(path=config),
        MCPProvider(connector=_memory_connector),
    )

    async with container:
        client = container.resolve(MCPClientContract)
        tools = await client.list_tools('memory')

    assert {tool.name for tool in tools} == {'echo'}


async def test_provider_serves_no_servers_without_a_config_file(tmp_path: Path) -> None:
    container = App().build(
        ConfigProvider(path=tmp_path / 'does-not-exist.yaml'),
        MCPProvider(connector=_memory_connector),
    )

    async with container:
        client = container.resolve(MCPClientContract)

        assert client.unavailable() == ()
        with pytest.raises(UnknownMCPServerError):
            await client.list_tools('memory')
