import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import Annotated, override

import pytest
from pydantic import Field

from app.core.tools import Tool, ToolOutput, ToolParams, ToolRegistry
from app.features.agents.services import AgentRunner, TurnInput
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


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(SleepTool())
    return registry


def make_turn(
    *,
    user_input: str = 'hello',
    model: str = 'test-model',
    prior_turns: tuple[Message, ...] = (),
    max_iterations: int = 10,
) -> TurnInput:
    return TurnInput(
        instructions='be helpful',
        user_input=user_input,
        model=model,
        prior_turns=prior_turns,
        max_iterations=max_iterations,
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
