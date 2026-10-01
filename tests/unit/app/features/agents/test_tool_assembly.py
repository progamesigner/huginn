from pathlib import Path
from typing import Annotated, override

import pytest
from pydantic import Field

from app.features.agents.config import AgentConfig, SkillsConfig, load_instructions
from app.features.agents.contracts import NamedTool, ToolHarvest, ToolSource
from app.features.agents.entities import ToolOutput, ToolParams, TurnScope, TypedTool
from app.features.agents.exceptions import AgentError, DuplicateToolNameError
from app.features.agents.services import build_tool_set

_SCOPE = TurnScope(actor='user:1', agent='test', channel='test', chat='1')


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
    unreachable: tuple[str, ...]

    def __init__(self, *names: str, unreachable: tuple[str, ...] = ()) -> None:
        self.named = tuple((name, EchoTool()) for name in names)
        self.seen = []
        self.unreachable = unreachable

    @override
    async def collect(self, agent: AgentConfig, scope: TurnScope) -> ToolHarvest:
        self.seen.append(agent)
        return ToolHarvest(tools=self.named, unreachable=self.unreachable)


def _agent(*, skills: SkillsConfig | None = None) -> AgentConfig:
    return AgentConfig(
        description='x',
        instructions='y',
        skills=skills or SkillsConfig(),
    )


async def test_build_tool_set_merges_every_source() -> None:
    tools, unreachable = await build_tool_set(
        (FakeSource('echo'), FakeSource('recall', 'skill')),
        _agent(),
        _SCOPE,
    )

    assert set(tools.tools) == {'echo', 'recall', 'skill'}
    assert unreachable == ()


async def test_build_tool_set_gathers_what_every_source_could_not_reach() -> None:
    _, unreachable = await build_tool_set(
        (
            FakeSource('echo', unreachable=('muninn',)),
            FakeSource('recall', unreachable=('skills',)),
        ),
        _agent(),
        _SCOPE,
    )

    assert unreachable == ('muninn', 'skills')


async def test_build_tool_set_drops_a_denied_tool() -> None:
    # NOTE: the denial narrows what the agent is configured with -- a caller may take
    # tools away from a turn, never add one the agent does not already have
    tools, _ = await build_tool_set(
        (FakeSource('echo', 'recall'),),
        _agent(),
        _SCOPE,
        frozenset({'recall'}),
    )

    assert set(tools.tools) == {'echo'}


async def test_build_tool_set_denying_an_absent_tool_adds_nothing() -> None:
    tools, _ = await build_tool_set(
        (FakeSource('echo'),),
        _agent(),
        _SCOPE,
        frozenset({'recall'}),
    )

    assert set(tools.tools) == {'echo'}


async def test_build_tool_set_is_empty_without_sources() -> None:
    tools, _ = await build_tool_set((), _agent(), _SCOPE)

    assert tools.tools == {}


async def test_build_tool_set_hands_every_source_the_agent() -> None:
    agent = _agent()
    sources = (FakeSource('echo'), FakeSource('recall'))

    await build_tool_set(sources, agent, _SCOPE)

    assert [source.seen for source in sources] == [[agent], [agent]]


async def test_build_tool_set_rejects_a_name_claimed_by_two_sources() -> None:
    with pytest.raises(DuplicateToolNameError, match='echo'):
        await build_tool_set(
            (FakeSource('echo'), FakeSource('echo')),
            _agent(),
            _SCOPE,
        )


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
