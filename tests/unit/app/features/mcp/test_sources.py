from collections.abc import Mapping
from typing import override

import pytest
from pydantic import JsonValue

from app.features.agents.config import AgentConfig, MCPRef
from app.features.mcp.sources import MCPToolSource
from lib.mcp import (
    MCPClientContract,
    MCPConnectionError,
    MCPToolInfo,
    UnknownMCPServerError,
)


class FakeClient(MCPClientContract):
    disconnected: tuple[str, ...]
    tools: dict[str, tuple[MCPToolInfo, ...]]

    def __init__(
        self,
        tools: dict[str, tuple[MCPToolInfo, ...]],
        disconnected: tuple[str, ...] = (),
    ) -> None:
        self.disconnected = disconnected
        self.tools = tools

    @override
    async def list_tools(self, server: str) -> tuple[MCPToolInfo, ...]:
        if server in self.disconnected:
            raise MCPConnectionError(f'{server}: not connected')

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

    @override
    def unavailable(self) -> tuple[str, ...]:
        return ()


def _info(name: str) -> MCPToolInfo:
    return MCPToolInfo(name=name, description='', input_schema={'type': 'object'})


def _agent(*refs: MCPRef) -> AgentConfig:
    return AgentConfig(description='x', instructions='y', mcp=refs)


async def test_collect_namespaces_and_applies_allowlist() -> None:
    client = FakeClient({'memory': (_info('recall'), _info('forget'))})
    agent = _agent(MCPRef(server='memory', allow=('recall',), inject={'user': 'swag'}))

    named = await MCPToolSource(client).collect(agent)

    assert [name for name, _ in named] == ['mcp__memory__recall']


async def test_collect_skips_unavailable_servers() -> None:
    client = FakeClient({'memory': (_info('recall'),)})
    agent = _agent(
        MCPRef(server='stock', allow=('quote',)),
        MCPRef(server='memory', allow=('recall',)),
    )

    named = await MCPToolSource(client).collect(agent)

    assert [name for name, _ in named] == ['mcp__memory__recall']


# NOTE: the live failure -- a configured server whose session had died raised
# MCPConnectionError, which this source did not catch, and every turn died on tool
# assembly before the model was ever asked anything
async def test_collect_skips_a_server_that_lost_its_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeClient({'memory': (_info('recall'),)}, disconnected=('muninn',))
    agent = _agent(
        MCPRef(server='muninn', allow=('recall_memory_notes',)),
        MCPRef(server='memory', allow=('recall',)),
    )

    with caplog.at_level('WARNING'):
        named = await MCPToolSource(client).collect(agent)

    assert [name for name, _ in named] == ['mcp__memory__recall']
    assert any('muninn' in message for message in caplog.messages)
