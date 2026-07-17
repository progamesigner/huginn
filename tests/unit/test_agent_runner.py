import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import Annotated, override

import pytest
from pydantic import Field
from structlog.contextvars import get_contextvars

from app.features.agents.exceptions import TurnTimeout
from app.features.agents.services import AgentRunner, TurnInput
from app.features.agents.tools import Tool, ToolOutput, ToolParams, ToolRegistry
from lib.llm import (
    CompletionChunk,
    CompletionRequest,
    FinishChunk,
    Message,
    TextChunk,
    ToolCallChunk,
    ToolResultSegment,
)


class FakeLLM:
    responses: list[list[CompletionChunk]]
    requests: list[CompletionRequest]

    def __init__(self, responses: Sequence[Sequence[CompletionChunk]]) -> None:
        self.responses = [list(response) for response in responses]
        self.requests = []

    async def stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionChunk]:
        self.requests.append(request)
        for chunk in self.responses.pop(0):
            yield chunk

    async def dispose(self) -> None:
        pass


class EchoParams(ToolParams):
    text: Annotated[str, Field()]


class EchoTool(Tool[EchoParams]):
    name = 'echo'
    description = 'Echo the given text back'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        return params.text


class SleepParams(ToolParams):
    seconds: Annotated[float, Field()]


class SleepTool(Tool[SleepParams]):
    name = 'sleep'
    description = 'Sleep for the given number of seconds'
    params = SleepParams

    @override
    async def run(self, params: SleepParams) -> ToolOutput:
        await asyncio.sleep(params.seconds)
        return 'slept'


class ProbeParams(ToolParams):
    pass


class ProbeTool(Tool[ProbeParams]):
    name = 'probe'
    description = 'Capture the contextvars bound during execution'
    params = ProbeParams

    seen: dict[str, object]

    def __init__(self) -> None:
        self.seen = {}

    @override
    async def run(self, params: ProbeParams) -> ToolOutput:
        self.seen = dict(get_contextvars())
        return 'probed'


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool)
    registry.register(SleepTool)
    return registry


def make_turn(
    *,
    user_input: str = 'hello',
    model: str = 'test-model',
    prior_turns: tuple[Message, ...] = (),
    max_iterations: int = 10,
    turn_timeout: float = 300.0,
    tool_timeout: float = 60.0,
) -> TurnInput:
    return TurnInput(
        instructions='be helpful',
        user_input=user_input,
        model=model,
        prior_turns=prior_turns,
        max_iterations=max_iterations,
        turn_timeout=turn_timeout,
        tool_timeout=tool_timeout,
    )


def tool_result(message: Message) -> ToolResultSegment:
    segment = message.segments[0]
    assert isinstance(segment, ToolResultSegment)
    return segment


async def test_run_returns_text_immediately_when_no_tool_calls() -> None:
    llm = FakeLLM(
        [
            [
                TextChunk(text='He'),
                TextChunk(text='llo'),
                FinishChunk(reason='stop'),
            ],
        ]
    )
    runner = AgentRunner(llm, make_registry())
    fragments: list[str] = []

    output = await runner.run(make_turn(), fragments.append)

    assert output.text == 'Hello'
    assert output.completed is True
    assert fragments == ['He', 'llo']
    assert len(llm.requests) == 1


async def test_run_executes_single_tool_call_then_finishes() -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(index=0, id='call_1', name='echo', arguments=''),
                ToolCallChunk(index=0, arguments='{"text"'),
                ToolCallChunk(index=0, arguments=': "hi"}'),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ]
    )
    runner = AgentRunner(llm, make_registry())

    output = await runner.run(make_turn(), lambda _: None)

    assert output.text == 'done'
    assert output.completed is True
    assert len(llm.requests) == 2

    result = tool_result(llm.requests[1].messages[-1])
    assert result.tool_call_id == 'call_1'
    assert result.content == 'hi'


async def test_run_executes_multiple_tool_calls_in_one_turn() -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='echo',
                    arguments='{"text": "a"}',
                ),
                ToolCallChunk(
                    index=1,
                    id='call_2',
                    name='echo',
                    arguments='{"text": "b"}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ]
    )
    runner = AgentRunner(llm, make_registry())

    output = await runner.run(make_turn(), lambda _: None)

    assert output.text == 'done'
    results = [tool_result(message) for message in llm.requests[1].messages[-2:]]
    assert {result.tool_call_id: result.content for result in results} == {
        'call_1': 'a',
        'call_2': 'b',
    }


async def test_run_executes_tool_calls_concurrently() -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='sleep',
                    arguments='{"seconds": 0.1}',
                ),
                ToolCallChunk(
                    index=1,
                    id='call_2',
                    name='sleep',
                    arguments='{"seconds": 0.1}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ]
    )
    runner = AgentRunner(llm, make_registry())

    started = time.monotonic()
    await runner.run(make_turn(), lambda _: None)
    elapsed = time.monotonic() - started

    assert elapsed < 0.18


async def test_run_iterates_across_multiple_turns_until_no_tool_calls() -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='echo',
                    arguments='{"text": "a"}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [
                ToolCallChunk(
                    index=0,
                    id='call_2',
                    name='echo',
                    arguments='{"text": "b"}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ]
    )
    runner = AgentRunner(llm, make_registry())

    output = await runner.run(make_turn(), lambda _: None)

    assert output.text == 'done'
    assert output.completed is True
    assert len(llm.requests) == 3
    assert len(llm.requests[2].messages) == len(llm.requests[0].messages) + 4


async def test_run_marks_incomplete_when_max_iterations_exhausted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def always_calls_tool() -> list[CompletionChunk]:
        return [
            ToolCallChunk(index=0, id='call_1', name='echo', arguments='{"text": "a"}'),
            FinishChunk(reason='tool_calls'),
        ]

    llm = FakeLLM([always_calls_tool(), always_calls_tool()])
    runner = AgentRunner(llm, make_registry())

    with caplog.at_level('WARNING'):
        output = await runner.run(make_turn(max_iterations=2), lambda _: None)

    assert output.completed is False
    assert any('max_iterations' in record.message for record in caplog.records)


async def test_run_reports_tool_timeout_as_error_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='sleep',
                    arguments='{"seconds": 1.0}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ]
    )
    runner = AgentRunner(llm, make_registry())

    with caplog.at_level('WARNING'):
        output = await runner.run(make_turn(tool_timeout=0.05), lambda _: None)

    assert output.text == 'done'
    assert output.completed is True

    result = tool_result(llm.requests[1].messages[-1])
    assert result.tool_call_id == 'call_1'
    assert result.error is True
    assert 'timed out' in result.content
    assert any('timed out' in record.message for record in caplog.records)


async def test_run_raises_turn_timeout_when_whole_turn_exceeds_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='sleep',
                    arguments='{"seconds": 1.0}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
        ]
    )
    runner = AgentRunner(llm, make_registry())

    with caplog.at_level('WARNING'), pytest.raises(TurnTimeout):
        await runner.run(make_turn(turn_timeout=0.05), lambda _: None)

    assert any('turn_timeout' in record.message for record in caplog.records)


async def test_run_records_steps_for_each_iteration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='echo',
                    arguments='{"text": "a"}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [
                ToolCallChunk(
                    index=0,
                    id='call_2',
                    name='echo',
                    arguments='{"text": "b"}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ]
    )
    runner = AgentRunner(llm, make_registry())

    with caplog.at_level('INFO'):
        output = await runner.run(make_turn(), lambda _: None)

    executions = [tool for step in output.steps for tool in step.tools]
    assert len(output.steps) == 3
    assert output.steps[-1].tools == ()
    assert [execution.name for execution in executions] == ['echo', 'echo']
    assert all(execution.ok for execution in executions)
    assert all(execution.duration >= 0 for execution in executions)
    assert any('finished' in record.message for record in caplog.records)
    assert all('"text"' not in record.message for record in caplog.records)


async def test_run_records_failed_tool_execution_in_steps() -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='sleep',
                    arguments='{"seconds": 1.0}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ]
    )
    runner = AgentRunner(llm, make_registry())

    output = await runner.run(make_turn(tool_timeout=0.05), lambda _: None)

    execution = output.steps[0].tools[0]
    assert execution.name == 'sleep'
    assert execution.ok is False


def probe_response() -> list[CompletionChunk]:
    return [
        ToolCallChunk(index=0, id='call_1', name='probe', arguments='{}'),
        FinishChunk(reason='tool_calls'),
    ]


async def test_run_binds_turn_id_during_tool_execution() -> None:
    llm = FakeLLM(
        [probe_response(), [TextChunk(text='done'), FinishChunk(reason='stop')]]
    )
    probe = ProbeTool()
    registry = ToolRegistry()
    registry.register(ProbeTool, factory=lambda: probe)
    runner = AgentRunner(llm, registry)

    await runner.run(make_turn(), lambda _: None)

    assert probe.seen.get('turn_id')
    assert 'turn_id' not in get_contextvars()


async def test_run_generates_unique_turn_id_per_turn() -> None:
    llm = FakeLLM(
        [
            probe_response(),
            [TextChunk(text='done'), FinishChunk(reason='stop')],
            probe_response(),
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ]
    )
    probe = ProbeTool()
    registry = ToolRegistry()
    registry.register(ProbeTool, factory=lambda: probe)
    runner = AgentRunner(llm, registry)

    await runner.run(make_turn(), lambda _: None)
    first = probe.seen['turn_id']
    await runner.run(make_turn(), lambda _: None)

    assert probe.seen['turn_id'] != first
