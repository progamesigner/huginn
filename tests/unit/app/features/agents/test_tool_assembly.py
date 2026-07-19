from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, override

import pytest
from pydantic import Field, JsonValue

from app.features.agents.config import AgentConfig, McpRef, ToolRef
from app.features.agents.exceptions import AgentError
from app.features.agents.services import build_tool_set, load_instructions
from app.features.tools.binding import ConfiguredTool
from app.features.tools.config import ToolConfig
from app.features.tools.exceptions import DuplicateToolError
from app.features.tools.registry import ToolRegistry
from app.features.tools.types import ToolOutput, ToolParams, TypedTool
from lib.mcp import McpClientContract, McpToolInfo, UnknownMcpServerError


class EchoParams(ToolParams):
    text: Annotated[str, Field()]


class EchoTool(TypedTool[EchoParams]):
    name = 'echo'
    description = 'Echo the text back'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        return params.text


class FakeClient(McpClientContract):
    tools: dict[str, tuple[McpToolInfo, ...]]

    def __init__(self, tools: dict[str, tuple[McpToolInfo, ...]]) -> None:
        self.tools = tools

    @override
    async def list_tools(self, server: str) -> tuple[McpToolInfo, ...]:
        if server not in self.tools:
            raise UnknownMcpServerError(server)

        return self.tools[server]

    @override
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        return 'called'


def _info(name: str) -> McpToolInfo:
    return McpToolInfo(name=name, description='', input_schema={'type': 'object'})


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool)
    return registry


def _agent(
    *, tools: tuple[ToolRef, ...] = (), mcp: tuple[McpRef, ...] = ()
) -> AgentConfig:
    return AgentConfig(description='x', instructions='y', tools=tools, mcp=mcp)


async def test_build_tool_set_combines_native_and_mcp_tools() -> None:
    configs: Mapping[str, ToolConfig] = {'echo': ToolConfig(kind='echo')}
    client = FakeClient({'memory': (_info('recall'),)})
    agent = _agent(
        tools=(ToolRef(name='echo'),),
        mcp=(McpRef(server='memory', allow=('recall',)),),
    )

    tools = await build_tool_set(_registry(), configs, client, agent)

    assert set(tools.tools) == {'echo', 'mcp__memory__recall'}


async def test_build_tool_set_wraps_native_tools_with_overlay() -> None:
    configs: Mapping[str, ToolConfig] = {'echo': ToolConfig(kind='echo')}
    client = FakeClient({})
    agent = _agent(tools=(ToolRef(name='echo', params={'text': 'hi'}),))

    tools = await build_tool_set(_registry(), configs, client, agent)

    assert isinstance(tools.tools['echo'], ConfiguredTool)


async def test_build_tool_set_rejects_name_collision_between_native_and_mcp() -> None:
    configs: Mapping[str, ToolConfig] = {'mcp__memory__recall': ToolConfig(kind='echo')}
    client = FakeClient({'memory': (_info('recall'),)})
    agent = _agent(
        tools=(ToolRef(name='mcp__memory__recall'),),
        mcp=(McpRef(server='memory', allow=('recall',)),),
    )

    with pytest.raises(DuplicateToolError, match='mcp__memory__recall'):
        await build_tool_set(_registry(), configs, client, agent)


def test_load_instructions_returns_inline_instructions() -> None:
    agent = AgentConfig(description='x', instructions='be helpful')

    assert load_instructions(agent) == 'be helpful'


def test_load_instructions_reads_instructions_file(tmp_path: Path) -> None:
    path = tmp_path / 'scout.txt'
    path.write_text('mounted instructions', encoding='utf-8')
    agent = AgentConfig(description='x', instructions_file=str(path))

    assert load_instructions(agent) == 'mounted instructions'


def test_load_instructions_raises_when_neither_is_set() -> None:
    agent = AgentConfig(description='x')

    with pytest.raises(AgentError):
        load_instructions(agent)
