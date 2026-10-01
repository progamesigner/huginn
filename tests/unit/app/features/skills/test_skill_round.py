from collections.abc import Mapping
from datetime import timedelta
from typing import override

from pydantic import JsonValue

from app.features.agents.config import AgentConfig, SkillsConfig
from app.features.agents.entities import TurnInput, TurnPrompt, TurnScope
from app.features.agents.services import AgentRunner, build_tool_set
from app.features.skills.config import SkillsSourceConfig
from app.features.skills.entities import Skill, SkillMeta
from app.features.skills.repositories import SkillRepository
from app.features.skills.sources.contracts import SkillSource
from app.features.skills.tools import SkillToolSource
from lib.llm import (
    FinishChunk,
    ModelCapabilities,
    TextChunk,
    ToolCallChunk,
    ToolResultSegment,
)
from lib.mcp import MCPClientContract, MCPToolInfo
from lib.temporal import ClockManager
from tests.helpers.clock import FakeClock
from tests.helpers.replies import RecordingReply
from tests.unit.app.features.agents.test_agent_runner import FakeLLM


class FakeSource(SkillSource):
    @override
    async def list(self, scope: Mapping[str, JsonValue]) -> tuple[SkillMeta, ...]:
        return (SkillMeta(name='triage', description='triage', path='s/triage.md'),)

    @override
    async def load(self, path: str, scope: Mapping[str, JsonValue]) -> Skill:
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
    tools, _ = await build_tool_set(
        (SkillToolSource(lambda: repository, SkillsSourceConfig()),),
        agent,
        TurnScope(actor='user:1', agent='test', channel='test', chat='1'),
    )
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
        TurnInput(
            instructions='be helpful',
            prompts=(TurnPrompt(text='help', actor='user:1'),),
            model='test-model',
            now=FakeClock().now(),
            channel='support_bot',
            capabilities=ModelCapabilities(),
        ),
        RecordingReply(),
    )

    result = llm.requests[1].messages[-1].segments[0]
    assert output.text == 'done'
    assert output.steps[0].tools[0].name == 'skill'
    assert isinstance(result, ToolResultSegment)
    assert 'Stay calm.' in result.content
