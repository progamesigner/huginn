from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Annotated, override

import pytest
from pydantic import Field, JsonValue

from app.features.agents.config import AgentConfig, MCPRef, SkillsConfig, ToolRef
from app.features.agents.exceptions import AgentError
from app.features.agents.services import build_tool_set, load_instructions
from app.features.skills.repositories import SkillRepository
from app.features.skills.sources.contracts import SkillSource
from app.features.skills.types import Skill, SkillMeta
from app.features.tools.binding import ConfiguredTool
from app.features.tools.config import ToolConfig
from app.features.tools.exceptions import DuplicateToolError
from app.features.tools.registry import ToolRegistry
from app.features.tools.types import ToolOutput, ToolParams, TypedTool
from lib.mcp import MCPClientContract, MCPToolInfo, UnknownMCPServerError
from tests.helpers.clock import FakeClock


class EchoParams(ToolParams):
    text: Annotated[str, Field()]


class EchoTool(TypedTool[EchoParams]):
    name = 'echo'
    description = 'Echo the text back'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        return params.text


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

    @override
    def unavailable(self) -> tuple[str, ...]:
        return ()


def _info(name: str) -> MCPToolInfo:
    return MCPToolInfo(name=name, description='', input_schema={'type': 'object'})


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool)
    return registry


def _agent(
    *,
    tools: tuple[ToolRef, ...] = (),
    mcp: tuple[MCPRef, ...] = (),
    skills: SkillsConfig | None = None,
) -> AgentConfig:
    return AgentConfig(
        description='x',
        instructions='y',
        tools=tools,
        mcp=mcp,
        skills=skills or SkillsConfig(),
    )


class FakeSkillSource(SkillSource):
    @override
    async def list(self) -> tuple[SkillMeta, ...]:
        return (
            SkillMeta(
                name='triage',
                description='triage incidents',
                path='s/triage.md',
            ),
            SkillMeta(name='report', description='write a report', path='s/report.md'),
        )

    @override
    async def load(self, path: str) -> Skill:
        return Skill(meta=await self._meta(path), body='body')

    async def _meta(self, path: str) -> SkillMeta:
        return next(meta for meta in await self.list() if meta.path == path)


def _skills() -> SkillRepository:
    return SkillRepository(FakeSkillSource(), FakeClock(), ttl=timedelta(minutes=15))


async def test_build_tool_set_adds_skill_tool_when_enabled() -> None:
    agent = _agent(skills=SkillsConfig(enabled=True))

    tools = await build_tool_set(_registry(), {}, FakeClient({}), agent, _skills())

    assert 'skill' in tools.tools


async def test_build_tool_set_omits_skill_tool_when_disabled() -> None:
    agent = _agent(skills=SkillsConfig(enabled=False))

    tools = await build_tool_set(_registry(), {}, FakeClient({}), agent, _skills())

    assert 'skill' not in tools.tools


async def test_build_tool_set_omits_skill_tool_without_a_repository() -> None:
    agent = _agent(skills=SkillsConfig(enabled=True))

    tools = await build_tool_set(_registry(), {}, FakeClient({}), agent)

    assert 'skill' not in tools.tools


async def test_build_tool_set_narrows_skills_to_the_allowlist() -> None:
    agent = _agent(skills=SkillsConfig(enabled=True, allow=('triage',)))

    tools = await build_tool_set(_registry(), {}, FakeClient({}), agent, _skills())

    description = tools.tools['skill'].description
    assert 'triage' in description
    assert 'report' not in description


async def test_build_tool_set_combines_native_and_mcp_tools() -> None:
    configs: Mapping[str, ToolConfig] = {'echo': ToolConfig(kind='echo')}
    client = FakeClient({'memory': (_info('recall'),)})
    agent = _agent(
        tools=(ToolRef(name='echo'),),
        mcp=(MCPRef(server='memory', allow=('recall',)),),
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
        mcp=(MCPRef(server='memory', allow=('recall',)),),
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
