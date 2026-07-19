from collections.abc import Mapping
from typing import override

from pydantic import JsonValue

from app.features.agents.config import MCPRef
from app.features.agents.services import collect_mcp_tools
from lib.mcp import MCPClientContract, MCPToolInfo, UnknownMCPServerError


class FakeClient(MCPClientContract):
    tools: dict[str, tuple[MCPToolInfo, ...]]

    def __init__(self, tools: dict[str, tuple[MCPToolInfo, ...]]) -> None:
        self.tools = tools

    @override
    async def list_tools(self, server: str) -> tuple[MCPToolInfo, ...]:
        if server not in self.tools:
            raise UnknownMCPServerError(server)

        return self.tools[server]

    @override
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        return 'called'


def _info(name: str) -> MCPToolInfo:
    return MCPToolInfo(name=name, description='', input_schema={'type': 'object'})


async def test_collect_namespaces_and_applies_allowlist() -> None:
    client = FakeClient({'memory': (_info('recall'), _info('forget'))})
    refs = [MCPRef(server='memory', allow=('recall',), inject={'user': 'swag'})]

    tools = await collect_mcp_tools(client, refs)

    assert [tool.name for tool in tools] == ['mcp__memory__recall']


async def test_collect_skips_unavailable_servers() -> None:
    client = FakeClient({'memory': (_info('recall'),)})
    refs = [
        MCPRef(server='stock', allow=('quote',)),
        MCPRef(server='memory', allow=('recall',)),
    ]

    tools = await collect_mcp_tools(client, refs)

    assert [tool.name for tool in tools] == ['mcp__memory__recall']
