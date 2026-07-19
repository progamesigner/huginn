from typing import Annotated

import pytest
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from lib.mcp import McpManager, McpServer

pytestmark = pytest.mark.integration


class LiveCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    memory_mcp_url: Annotated[str | None, Field()] = None
    memory_mcp_token: Annotated[SecretStr | None, Field()] = None


def require_memory_server() -> McpServer:
    credentials = LiveCredentials()
    if credentials.memory_mcp_url is None or credentials.memory_mcp_token is None:
        pytest.skip('MEMORY_MCP_URL / MEMORY_MCP_TOKEN not set')

    token = credentials.memory_mcp_token.get_secret_value()
    return McpServer(
        name='memory',
        url=credentials.memory_mcp_url,
        headers={'Authorization': f'Bearer {token}'},
    )


async def test_lists_and_calls_a_read_only_tool() -> None:
    manager = McpManager([require_memory_server()])

    await manager.start()
    try:
        assert list(manager.connections) == ['memory']

        tools = await manager.list_tools('memory')
        names = {tool.name for tool in tools}
        assert 'list_memory_notes' in names

        # scope is injected by the harness in production; supply it directly here
        output = await manager.call_tool(
            'memory',
            'list_memory_notes',
            {'agent': 'default', 'user': 'default'},
        )
        assert isinstance(output, str)
    finally:
        await manager.dispose()
