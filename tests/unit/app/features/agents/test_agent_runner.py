import asyncio
import time
from base64 import b64decode, b64encode
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Annotated, override

import pytest
from pydantic import Field, JsonValue
from structlog.contextvars import bound_contextvars, get_contextvars

from app.contracts.channels import AttachmentKind, ReplyArtifact, ToolPhase
from app.features.agents.contracts import (
    ToolOutput,
    ToolParams,
    ToolResult,
    TypedTool,
)
from app.features.agents.entities import (
    PromptAttachment,
    TurnBudget,
    TurnInput,
    TurnPrompt,
)
from app.features.agents.exceptions import MalformedToolCallError
from app.features.agents.services import AgentRunner
from app.features.agents.tooling import ToolSet
from lib.llm import (
    CompletionChunk,
    CompletionRequest,
    DocumentSegment,
    FinishChunk,
    ImageSegment,
    Message,
    ModelCapabilities,
    TextChunk,
    TextSegment,
    ToolCallChunk,
    ToolResultSegment,
)
from lib.temporal import ClockManager
from tests.helpers.clock import FakeClock
from tests.helpers.logging import make_logger
from tests.helpers.replies import RecordingReply

_VISION = ModelCapabilities(vision=True)


class FakeLLM:
    responses: list[list[CompletionChunk]]
    requests: list[CompletionRequest]

    def __init__(self, responses: Sequence[Sequence[CompletionChunk]]) -> None:
        self.responses = [list(response) for response in responses]
        self.requests = []

    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[CompletionChunk]:
        self.requests.append(request)
        for chunk in self.responses.pop(0):
            yield chunk

    async def dispose(self) -> None:
        pass


class StallingLLM:
    fragments: tuple[str, ...]
    requests: list[CompletionRequest]

    def __init__(self, *fragments: str) -> None:
        self.fragments = fragments
        self.requests = []

    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[CompletionChunk]:
        self.requests.append(request)
        for fragment in self.fragments:
            yield TextChunk(text=fragment)
        await asyncio.sleep(60)

    async def dispose(self) -> None:
        pass


class EchoParams(ToolParams):
    text: Annotated[str, Field()]


class EchoTool(TypedTool[EchoParams]):
    name = 'echo'
    description = 'Echo the given text back'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        return params.text


class SleepParams(ToolParams):
    seconds: Annotated[float, Field()]


class SleepTool(TypedTool[SleepParams]):
    name = 'sleep'
    description = 'Sleep for the given number of seconds'
    params = SleepParams

    @override
    async def run(self, params: SleepParams) -> ToolOutput:
        await asyncio.sleep(params.seconds)
        return 'slept'


class ProbeParams(ToolParams):
    pass


class ProbeTool(TypedTool[ProbeParams]):
    name = 'probe'
    description = 'Capture the contextvars bound during execution'
    params = ProbeParams

    seen: dict[str, JsonValue]

    def __init__(self) -> None:
        self.seen = {}

    @override
    async def run(self, params: ProbeParams) -> ToolOutput:
        self.seen = dict(get_contextvars())
        return 'probed'


class AdvancingTool(TypedTool[ProbeParams]):
    name = 'advance'
    description = 'Advance the injected clock while executing'
    params = ProbeParams

    clock: FakeClock
    step: timedelta

    def __init__(self, clock: FakeClock, step: timedelta) -> None:
        self.clock = clock
        self.step = step

    @override
    async def run(self, params: ProbeParams) -> ToolOutput:
        self.clock.advance(self.step)
        return 'advanced'


def make_toolset() -> ToolSet:
    tools = ToolSet(logger=make_logger())
    tools.add('echo', EchoTool())
    tools.add('sleep', SleepTool())
    return tools


def make_turn(
    *,
    prompts: tuple[TurnPrompt, ...] = (TurnPrompt(text='hello', actor='user:1'),),
    model: str = 'test-model',
    now: datetime = datetime(2026, 1, 1, tzinfo=UTC),
    channel: str = 'support_bot',
    capabilities: ModelCapabilities = _VISION,
    prior_turns: tuple[Message, ...] = (),
    max_completion_tokens: int | None = None,
    max_iterations: int = 10,
    turn_timeout: float = 300.0,
    tool_timeout: float = 60.0,
    min_iteration_budget: float = 0.0,
) -> TurnInput:
    return TurnInput(
        instructions='be helpful',
        prompts=prompts,
        model=model,
        now=now,
        channel=channel,
        capabilities=capabilities,
        prior_turns=prior_turns,
        max_completion_tokens=max_completion_tokens,
        budget=TurnBudget(
            max_iterations=max_iterations,
            turn_timeout=turn_timeout,
            tool_timeout=tool_timeout,
            min_iteration_budget=min_iteration_budget,
        ),
    )


def attachment(
    kind: AttachmentKind = 'image',
    *,
    data: bytes = b'\xff\xd8\xff\xe0jpeg',
    mime: str = 'image/jpeg',
    hint: str = '[image: image/jpeg 8x8]',
) -> PromptAttachment:
    return PromptAttachment(
        kind=kind,
        mime=mime,
        reference='file-1',
        unique='unique-1',
        data=data,
        hint=hint,
    )


def tool_result(message: Message) -> ToolResultSegment:
    segment = message.segments[0]
    assert isinstance(segment, ToolResultSegment)
    return segment


def answering() -> FakeLLM:
    return FakeLLM([[TextChunk(text='done'), FinishChunk(reason='stop')]])


def discard() -> RecordingReply:
    return RecordingReply()


async def test_run_returns_text_immediately_when_no_tool_calls() -> None:
    llm = FakeLLM(
        [
            [
                TextChunk(text='He'),
                TextChunk(text='llo'),
                FinishChunk(reason='stop'),
            ],
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())
    sink = RecordingReply()

    output = await runner.run(make_turn(), sink)

    assert output.text == 'Hello'
    assert output.completed is True
    assert sink.fragments == ['He', 'llo']
    assert len(llm.requests) == 1


async def test_run_sends_the_token_limit_on_every_tool_loop_iteration() -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='echo',
                    arguments='{"text": "hi"}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ],
    )

    await AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger()).run(
        make_turn(max_completion_tokens=4096),
        discard(),
    )

    assert [request.max_completion_tokens for request in llm.requests] == [4096, 4096]


def breakpoints(request: CompletionRequest) -> list[int]:
    return [
        index for index, message in enumerate(request.messages) if message.cacheable
    ]


async def test_run_marks_the_instructions_as_the_static_breakpoint() -> None:
    llm = answering()

    await AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger()).run(
        make_turn(),
        discard(),
    )

    system = llm.requests[0].messages[0]
    assert system.role == 'system'
    assert system.cacheable is True


async def test_run_rolls_the_second_breakpoint_onto_the_newest_message() -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='echo',
                    arguments='{"text": "hi"}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ],
    )

    await AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger()).run(
        make_turn(),
        discard(),
    )

    first, second = llm.requests
    # NOTE: two breakpoints per request, never more -- Anthropic allows four and
    # a tool loop would blow past that if each iteration left one behind
    assert breakpoints(first) == [0, len(first.messages) - 1]
    assert breakpoints(second) == [0, len(second.messages) - 1]
    assert len(second.messages) > len(first.messages)
    assert second.messages[-1].role == 'tool'


async def test_run_does_not_leave_a_breakpoint_on_a_replayed_prior_turn() -> None:
    prior = (
        Message(role='user', segments=(TextSegment(text='earlier'),)),
        Message(role='assistant', segments=(TextSegment(text='sure'),)),
    )
    llm = answering()

    await AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger()).run(
        make_turn(prior_turns=prior),
        discard(),
    )

    assert [message.cacheable for message in prior] == [False, False]
    assert breakpoints(llm.requests[0]) == [0, 3]


# NOTE: a compacted conversation replays its summary already marked, so the
# static and rolling breakpoints land on top of it -- Anthropic allows four
async def test_run_stays_within_the_breakpoint_budget_with_a_replayed_summary() -> None:
    prior = (
        Message(
            role='user',
            segments=(TextSegment(text='summary so far'),),
            cacheable=True,
        ),
        Message(role='assistant', segments=(TextSegment(text='sure'),)),
    )
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='echo',
                    arguments='{"text": "hi"}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ],
    )

    await AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger()).run(
        make_turn(prior_turns=prior),
        discard(),
    )

    assert [len(breakpoints(request)) for request in llm.requests] == [3, 3]


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
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())

    output = await runner.run(make_turn(), discard())

    assert output.text == 'done'
    assert output.completed is True
    assert len(llm.requests) == 2

    result = tool_result(llm.requests[1].messages[-1])
    assert result.tool_call_id == 'call_1'
    assert result.content == 'hi'


async def test_run_tells_the_stream_when_a_tool_starts_and_finishes() -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='echo',
                    arguments='{"text": "hi"}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())
    sink = RecordingReply()

    await runner.run(make_turn(), sink)

    assert sink.tools == [('echo', 'started'), ('echo', 'finished')]


async def test_run_reports_a_tool_that_answered_with_an_error_as_failed() -> None:
    # NOTE: a tool failure is a result the model gets to read, not an exception -- the
    # stream still has to say the step went wrong so a client can show it
    llm = FakeLLM(
        [
            [
                ToolCallChunk(index=0, id='call_1', name='ghost', arguments='{}'),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())
    sink = RecordingReply()

    await runner.run(make_turn(), sink)

    assert sink.tools == [('ghost', 'started'), ('ghost', 'failed')]


# NOTE: the same class of gateway defect that omits the completed output list also skips
# the item event, and an empty call id sent back correlates to nothing upstream
async def test_run_rejects_a_tool_call_that_arrived_without_a_call_id() -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(index=0, arguments='{"text": "hi"}'),
                FinishChunk(reason='tool_calls'),
            ],
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())
    sink = RecordingReply()

    with pytest.raises(MalformedToolCallError):
        await runner.run(make_turn(), sink)

    assert sink.tools == []
    assert len(llm.requests) == 1


async def test_run_rejects_only_the_call_id_and_not_an_unnamed_tool() -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(index=0, id='call_1', arguments='{}'),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())
    sink = RecordingReply()

    output = await runner.run(make_turn(), sink)

    assert output.text == 'done'
    assert sink.tools == [('', 'started'), ('', 'failed')]


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
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())

    output = await runner.run(make_turn(), discard())

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
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())

    started = time.monotonic()
    await runner.run(make_turn(), discard())
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
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())

    output = await runner.run(make_turn(), discard())

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
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())

    with caplog.at_level('WARNING'):
        output = await runner.run(make_turn(max_iterations=2), discard())

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
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())

    with caplog.at_level('WARNING'):
        output = await runner.run(make_turn(tool_timeout=0.05), discard())

    assert output.text == 'done'
    assert output.completed is True

    result = tool_result(llm.requests[1].messages[-1])
    assert result.tool_call_id == 'call_1'
    assert result.error is True
    assert 'timed out' in result.content
    assert any('timed out' in record.message for record in caplog.records)


# NOTE: this is the shape that broke a live round -- the budget expired while the final
# answer was streaming, the runner raised, and the whole turn (prompts included) rolled
# out of the transcript. An expired turn keeps what it said.
async def test_run_delivers_the_partial_answer_when_the_turn_budget_expires(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = StallingLLM('half an ')
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())
    sink = RecordingReply()

    with caplog.at_level('WARNING'):
        output = await runner.run(make_turn(turn_timeout=0.05), sink)

    assert output.text == 'half an '
    assert output.completed is False
    assert sink.fragments == ['half an ']
    assert any('turn_timeout' in record.message for record in caplog.records)


async def test_run_keeps_the_streamed_text_when_the_budget_expires_in_a_tool() -> None:
    llm = FakeLLM(
        [
            [
                TextChunk(text='let me look'),
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='sleep',
                    arguments='{"seconds": 5.0}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())

    output = await runner.run(make_turn(turn_timeout=0.05), discard())

    assert output.text == 'let me look'
    assert output.completed is False


async def test_run_does_not_open_an_iteration_it_cannot_afford(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='sleep',
                    arguments='{"seconds": 0.1}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='never asked'), FinishChunk(reason='stop')],
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())

    with caplog.at_level('WARNING'):
        output = await runner.run(
            make_turn(turn_timeout=5.0, min_iteration_budget=4.95),
            discard(),
        )

    assert output.completed is False
    assert len(llm.requests) == 1
    assert any('turn_timeout' in record.message for record in caplog.records)


async def test_run_always_opens_the_first_iteration_however_small_the_budget() -> None:
    llm = answering()
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())

    output = await runner.run(
        make_turn(turn_timeout=300.0, min_iteration_budget=600.0),
        discard(),
    )

    assert output.text == 'done'
    assert output.completed is True
    assert len(llm.requests) == 1


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
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())

    with caplog.at_level('INFO'):
        output = await runner.run(make_turn(), discard())

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
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())

    output = await runner.run(make_turn(tool_timeout=0.05), discard())

    execution = output.steps[0].tools[0]
    assert execution.name == 'sleep'
    assert execution.ok is False


async def test_tool_duration_is_measured_by_the_injected_clock() -> None:
    clock = FakeClock()
    tools = ToolSet(logger=make_logger())
    tools.add('advance', AdvancingTool(clock, timedelta(seconds=2)))
    llm = FakeLLM(
        [
            [
                ToolCallChunk(index=0, id='call_1', name='advance', arguments='{}'),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ],
    )
    runner = AgentRunner(llm, tools, clock, logger=make_logger())

    output = await runner.run(make_turn(), discard())

    execution = output.steps[0].tools[0]
    assert execution.name == 'advance'
    assert execution.duration == 2.0


def probe_response() -> list[CompletionChunk]:
    return [
        ToolCallChunk(index=0, id='call_1', name='probe', arguments='{}'),
        FinishChunk(reason='tool_calls'),
    ]


async def test_run_binds_turn_id_during_tool_execution() -> None:
    llm = FakeLLM(
        [probe_response(), [TextChunk(text='done'), FinishChunk(reason='stop')]],
    )
    probe = ProbeTool()
    tools = ToolSet(logger=make_logger())
    tools.add('probe', probe)
    runner = AgentRunner(llm, tools, ClockManager(), logger=make_logger())

    await runner.run(make_turn(), discard())

    assert probe.seen.get('turn_id')
    assert 'turn_id' not in get_contextvars()


async def test_run_generates_unique_turn_id_per_turn() -> None:
    llm = FakeLLM(
        [
            probe_response(),
            [TextChunk(text='done'), FinishChunk(reason='stop')],
            probe_response(),
            [TextChunk(text='done'), FinishChunk(reason='stop')],
        ],
    )
    probe = ProbeTool()
    tools = ToolSet(logger=make_logger())
    tools.add('probe', probe)
    runner = AgentRunner(llm, tools, ClockManager(), logger=make_logger())

    await runner.run(make_turn(), discard())
    first = probe.seen['turn_id']
    await runner.run(make_turn(), discard())

    assert probe.seen['turn_id'] != first


async def test_an_attached_image_reaches_the_model_beside_its_caption() -> None:
    llm = answering()
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())
    prompt = TurnPrompt(
        text='what is this?',
        actor='user:1',
        attachments=(attachment(),),
    )

    await runner.run(make_turn(prompts=(prompt,)), discard())

    # NOTE: segments[0] is the current-time stamp every user message carries
    segments = llm.requests[0].messages[-1].segments[1:]
    assert [type(segment) for segment in segments] == [TextSegment, ImageSegment]
    image = segments[1]
    assert isinstance(image, ImageSegment)
    assert image.media_type == 'image/jpeg'
    assert b64decode(image.data) == b'\xff\xd8\xff\xe0jpeg'


async def test_a_photo_without_a_caption_is_still_a_message() -> None:
    llm = answering()
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())
    prompt = TurnPrompt(text='', actor='user:1', attachments=(attachment(),))

    await runner.run(make_turn(prompts=(prompt,)), discard())

    # NOTE: the old shape built the message from text alone, so a caption-less photo
    # reached the model as an empty turn and answered nothing. segments[0] is the
    # current-time stamp every user message carries.
    segments = llm.requests[0].messages[-1].segments[1:]
    assert [type(segment) for segment in segments] == [ImageSegment]


async def test_a_model_without_vision_gets_the_image_hint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = answering()
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())
    prompt = TurnPrompt(text='', actor='user:1', attachments=(attachment(),))

    with caplog.at_level('INFO'):
        await runner.run(
            make_turn(prompts=(prompt,)).model_copy(
                update={'capabilities': ModelCapabilities()},
            ),
            discard(),
        )

    assert llm.requests[0].messages[-1].segments[1:] == (
        TextSegment(text='[image: image/jpeg 8x8]'),
    )
    assert 'kind=image reason=no_vision' in caplog.text


@pytest.mark.parametrize('documents', [True, False])
async def test_a_pdf_reaches_the_model_as_a_document_only_when_it_reads_them(
    documents: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = answering()
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())
    pdf = attachment(
        'document',
        data=b'%PDF',
        mime='application/pdf',
        hint='[pdf]',
    ).model_copy(
        update={'name': 'report.pdf', 'transcript': '[pdf] (1 pages)\nfirst words'},
    )
    prompt = TurnPrompt(text='', actor='user:1', attachments=(pdf,))

    with caplog.at_level('INFO'):
        await runner.run(
            make_turn(prompts=(prompt,)).model_copy(
                update={'capabilities': ModelCapabilities(documents=documents)},
            ),
            discard(),
        )

    segments = llm.requests[0].messages[-1].segments[1:]
    if documents:
        assert segments == (
            DocumentSegment(
                media_type='application/pdf',
                data=b64encode(b'%PDF').decode('ascii'),
                name='report.pdf',
            ),
        )
    else:
        assert segments == (TextSegment(text='[pdf] (1 pages)\nfirst words'),)
        assert 'kind=document reason=no_documents' in caplog.text


async def test_a_transcribed_voice_note_arrives_as_what_it_said() -> None:
    llm = answering()
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())
    voice = attachment('audio', mime='audio/ogg', hint='[audio: audio/ogg 12s]')
    prompt = TurnPrompt(
        text='',
        actor='user:1',
        attachments=(voice.model_copy(update={'transcript': 'remind me at six'}),),
    )

    await runner.run(make_turn(prompts=(prompt,)), discard())

    # NOTE: segments[0] is the current-time stamp every user message carries
    assert llm.requests[0].messages[-1].segments[1:] == (
        TextSegment(text='remind me at six'),
    )


async def test_an_attachment_the_model_cannot_take_arrives_as_its_hint() -> None:
    llm = answering()
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())
    voice = attachment('audio', mime='audio/ogg', hint='[audio: audio/ogg 12s]')
    prompt = TurnPrompt(text='', actor='user:1', attachments=(voice,))

    await runner.run(make_turn(prompts=(prompt,)), discard())

    # NOTE: segments[0] is the current-time stamp every user message carries
    assert llm.requests[0].messages[-1].segments[1:] == (
        TextSegment(text='[audio: audio/ogg 12s]'),
    )


async def test_a_prompt_with_nothing_in_it_is_not_a_message() -> None:
    llm = answering()
    runner = AgentRunner(llm, make_toolset(), ClockManager(), logger=make_logger())
    prompt = TurnPrompt(text='', actor='user:1')

    await runner.run(make_turn(prompts=(prompt,)), discard())

    assert [message.role for message in llm.requests[0].messages] == ['system']


# NOTE: a channel opens the turn before the runner sees it and needs the id for its own
# failure boundary, so the runner joins that turn instead of naming it a second time
async def test_run_joins_a_turn_id_its_caller_already_bound() -> None:
    llm = FakeLLM(
        [probe_response(), [TextChunk(text='done'), FinishChunk(reason='stop')]],
    )
    probe = ProbeTool()
    tools = ToolSet(logger=make_logger())
    tools.add('probe', probe)
    runner = AgentRunner(llm, tools, ClockManager(), logger=make_logger())

    with bound_contextvars(turn_id='opened-by-the-channel'):
        await runner.run(make_turn(), discard())

    assert probe.seen['turn_id'] == 'opened-by-the-channel'


ARTIFACT = ReplyArtifact(
    blob='0190a1b2-0000-7000-8000-000000000001',
    kind='image',
    mime='image/png',
    size=4,
    data=b'\x89PNG',
    name='chart.png',
)


class DrawTool(TypedTool[ProbeParams]):
    name = 'draw'
    description = 'Produce an image'
    params = ProbeParams

    @override
    async def run(self, params: ProbeParams) -> ToolOutput:
        return ToolResult(text='Drew a chart.', artifacts=(ARTIFACT,))


class OrderedReply(RecordingReply):
    llm: FakeLLM
    log: list[str]

    def __init__(self, llm: FakeLLM) -> None:
        super().__init__()
        self.llm = llm
        self.log = []

    @override
    async def on_artifact(self, artifact: ReplyArtifact) -> None:
        await super().on_artifact(artifact)
        self.log.append(f'artifact after {len(self.llm.requests)} requests')

    @override
    async def on_tool(self, name: str, phase: ToolPhase) -> None:
        await super().on_tool(name, phase)
        self.log.append(f'tool {phase}')


def drawing() -> FakeLLM:
    return FakeLLM(
        [
            [
                ToolCallChunk(index=0, id='call_1', name='draw', arguments='{}'),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='here it is'), FinishChunk(reason='stop')],
        ],
    )


async def test_run_hands_an_artifact_to_the_reader_after_its_tool_finishes() -> None:
    llm = drawing()
    tools = ToolSet(logger=make_logger())
    tools.add('draw', DrawTool())
    sink = OrderedReply(llm)

    await AgentRunner(llm, tools, ClockManager(), logger=make_logger()).run(
        make_turn(),
        sink,
    )

    assert sink.artifacts == [ARTIFACT]
    assert sink.log == [
        'tool started',
        'tool finished',
        'artifact after 1 requests',
    ]


async def test_run_shows_the_model_only_the_text_of_an_artifact_tool() -> None:
    llm = drawing()
    tools = ToolSet(logger=make_logger())
    tools.add('draw', DrawTool())

    await AgentRunner(llm, tools, ClockManager(), logger=make_logger()).run(
        make_turn(),
        discard(),
    )

    result = tool_result(llm.requests[1].messages[-1])
    assert result.content == 'Drew a chart.'
    assert 'PNG' not in llm.requests[1].model_dump_json()
