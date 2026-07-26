from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, override

import pytest
from pydantic import Field

from app.features.agents.config import AgentConfig, SkillsConfig
from app.features.agents.contracts import NamedTool, ToolSource
from app.features.agents.exceptions import AgentError, DuplicateToolNameError
from app.features.agents.services import build_tool_set, load_instructions
from app.features.agents.types import ToolOutput, ToolParams, TypedTool


class EchoParams(ToolParams):
    text: Annotated[str, Field()]


class EchoTool(TypedTool[EchoParams]):
    name = 'echo'
    description = 'Echo the text back'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        return params.text


class FakeSource(ToolSource):
    named: tuple[NamedTool, ...]
    seen: list[AgentConfig]

    def __init__(self, *names: str) -> None:
        self.named = tuple((name, EchoTool()) for name in names)
        self.seen = []

    @override
    async def collect(self, agent: AgentConfig) -> Sequence[NamedTool]:
        self.seen.append(agent)
        return self.named


def _agent(*, skills: SkillsConfig | None = None) -> AgentConfig:
    return AgentConfig(
        description='x',
        instructions='y',
        skills=skills or SkillsConfig(),
    )


async def test_build_tool_set_merges_every_source() -> None:
    tools = await build_tool_set(
        (FakeSource('echo'), FakeSource('recall', 'skill')),
        _agent(),
    )

    assert set(tools.tools) == {'echo', 'recall', 'skill'}


async def test_build_tool_set_is_empty_without_sources() -> None:
    tools = await build_tool_set((), _agent())

    assert tools.tools == {}


async def test_build_tool_set_hands_every_source_the_agent() -> None:
    agent = _agent()
    sources = (FakeSource('echo'), FakeSource('recall'))

    await build_tool_set(sources, agent)

    assert [source.seen for source in sources] == [[agent], [agent]]


async def test_build_tool_set_rejects_a_name_claimed_by_two_sources() -> None:
    with pytest.raises(DuplicateToolNameError, match='echo'):
        await build_tool_set((FakeSource('echo'), FakeSource('echo')), _agent())


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
