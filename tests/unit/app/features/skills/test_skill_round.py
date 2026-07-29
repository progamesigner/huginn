from collections.abc import Mapping
from datetime import timedelta
from typing import override

from pydantic import JsonValue

from app.features.agents.config import AgentConfig, SkillsConfig
from app.features.agents.services import AgentRunner, build_tool_set
from app.features.agents.types import TurnInput
from app.features.skills.repositories import SkillRepository
from app.features.skills.sources.contracts import SkillSource
from app.features.skills.tools import SkillToolSource
from app.features.skills.types import Skill, SkillMeta
from lib.llm import FinishChunk, TextChunk, ToolCallChunk, ToolResultSegment
from lib.mcp import MCPClientContract, MCPToolInfo
from lib.temporal import ClockManager
from tests.helpers.clock import FakeClock
from tests.unit.app.features.agents.test_agent_runner import FakeLLM, discard


class FakeSource(SkillSource):
    @override
    async def list(self) -> tuple[SkillMeta, ...]:
        return (SkillMeta(name='triage', description='triage', path='s/triage.md'),)

    @override
    async def load(self, path: str) -> Skill:
        return Skill(
            meta=SkillMeta(name='triage', description='triage', path=path),
            body='# Triage\n\n1. Stay calm.',
        )


class NoMCP(MCPClientContract):
    @override
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        return ''

    @override
    async def list_tools(self, server: str) -> tuple[MCPToolInfo, ...]:
        return ()

    @override
    def unavailable(self) -> tuple[str, ...]:
        return ()


async def test_agent_loads_a_skill_through_the_tool_loop() -> None:
    repository = SkillRepository(FakeSource(), FakeClock(), ttl=timedelta(minutes=15))
    agent = AgentConfig(
        description='x',
        instructions='y',
        skills=SkillsConfig(enabled=True),
    )
    tools = await build_tool_set((SkillToolSource(lambda: repository),), agent)
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='c1',
                    name='skill',
                    arguments='{"name": "triage"}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ],
    )
    runner = AgentRunner(llm, tools, ClockManager())

    output = await runner.run(
        TurnInput(instructions='be helpful', prompts=('help',), model='test-model'),
        discard,
    )

    result = llm.requests[1].messages[-1].segments[0]
    assert output.text == 'done'
    assert output.steps[0].tools[0].name == 'skill'
    assert isinstance(result, ToolResultSegment)
    assert 'Stay calm.' in result.content
