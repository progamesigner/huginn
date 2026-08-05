import asyncio
import time
from base64 import b64decode
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Annotated, override

import pytest
from pydantic import Field, JsonValue
from structlog.contextvars import bound_contextvars, get_contextvars

from app.contracts.channels import AttachmentKind
from app.features.agents.exceptions import TurnTimeoutError
from app.features.agents.services import AgentRunner
from app.features.agents.types import (
    PromptAttachment,
    ToolOutput,
    ToolParams,
    ToolSet,
    TurnInput,
    TurnPrompt,
    TypedTool,
)
from lib.llm import (
    CompletionChunk,
    CompletionRequest,
    FinishChunk,
    ImageSegment,
    Message,
    TextChunk,
    TextSegment,
    ToolCallChunk,
    ToolResultSegment,
)
from lib.temporal import ClockManager
from tests.helpers.clock import FakeClock
from tests.helpers.replies import RecordingReply


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
    tools = ToolSet()
    tools.add('echo', EchoTool())
    tools.add('sleep', SleepTool())
    return tools


def make_turn(
    *,
    prompts: tuple[TurnPrompt, ...] = (TurnPrompt(text='hello', actor='user:1'),),
    model: str = 'test-model',
    now: datetime = datetime(2026, 1, 1, tzinfo=UTC),
    channel: str = 'support_bot',
    prior_turns: tuple[Message, ...] = (),
    max_completion_tokens: int | None = None,
    max_iterations: int = 10,
    turn_timeout: float = 300.0,
    tool_timeout: float = 60.0,
) -> TurnInput:
    return TurnInput(
        instructions='be helpful',
        prompts=prompts,
        model=model,
        now=now,
        channel=channel,
        prior_turns=prior_turns,
        max_completion_tokens=max_completion_tokens,
        max_iterations=max_iterations,
        turn_timeout=turn_timeout,
        tool_timeout=tool_timeout,
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
    runner = AgentRunner(llm, make_toolset(), ClockManager())
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

    await AgentRunner(llm, make_toolset(), ClockManager()).run(
        make_turn(max_completion_tokens=4096),
        discard(),
    )

    assert [request.max_completion_tokens for request in llm.requests] == [4096, 4096]


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
    runner = AgentRunner(llm, make_toolset(), ClockManager())

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
    runner = AgentRunner(llm, make_toolset(), ClockManager())
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
    runner = AgentRunner(llm, make_toolset(), ClockManager())
    sink = RecordingReply()

    await runner.run(make_turn(), sink)

    assert sink.tools == [('ghost', 'started'), ('ghost', 'failed')]


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
    runner = AgentRunner(llm, make_toolset(), ClockManager())

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
    runner = AgentRunner(llm, make_toolset(), ClockManager())

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
    runner = AgentRunner(llm, make_toolset(), ClockManager())

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
    runner = AgentRunner(llm, make_toolset(), ClockManager())

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
    runner = AgentRunner(llm, make_toolset(), ClockManager())

    with caplog.at_level('WARNING'):
        output = await runner.run(make_turn(tool_timeout=0.05), discard())

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
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager())

    with caplog.at_level('WARNING'), pytest.raises(TurnTimeoutError):
        await runner.run(make_turn(turn_timeout=0.05), discard())

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
        ],
    )
    runner = AgentRunner(llm, make_toolset(), ClockManager())

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
    runner = AgentRunner(llm, make_toolset(), ClockManager())

    output = await runner.run(make_turn(tool_timeout=0.05), discard())

    execution = output.steps[0].tools[0]
    assert execution.name == 'sleep'
    assert execution.ok is False


async def test_tool_duration_is_measured_by_the_injected_clock() -> None:
    clock = FakeClock()
    tools = ToolSet()
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
    runner = AgentRunner(llm, tools, clock)

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
    tools = ToolSet()
    tools.add('probe', probe)
    runner = AgentRunner(llm, tools, ClockManager())

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
    tools = ToolSet()
    tools.add('probe', probe)
    runner = AgentRunner(llm, tools, ClockManager())

    await runner.run(make_turn(), discard())
    first = probe.seen['turn_id']
    await runner.run(make_turn(), discard())

    assert probe.seen['turn_id'] != first


async def test_an_attached_image_reaches_the_model_beside_its_caption() -> None:
    llm = answering()
    runner = AgentRunner(llm, make_toolset(), ClockManager())
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
    runner = AgentRunner(llm, make_toolset(), ClockManager())
    prompt = TurnPrompt(text='', actor='user:1', attachments=(attachment(),))

    await runner.run(make_turn(prompts=(prompt,)), discard())

    # NOTE: the old shape built the message from text alone, so a caption-less photo
    # reached the model as an empty turn and answered nothing. segments[0] is the
    # current-time stamp every user message carries.
    segments = llm.requests[0].messages[-1].segments[1:]
    assert [type(segment) for segment in segments] == [ImageSegment]


async def test_a_transcribed_voice_note_arrives_as_what_it_said() -> None:
    llm = answering()
    runner = AgentRunner(llm, make_toolset(), ClockManager())
    voice = attachment('audio', mime='audio/ogg', hint='[audio: audio/ogg 12s]')
    prompt = TurnPrompt(
        text='',
        actor='user:1',
        attachments=(replace(voice, transcript='remind me at six'),),
    )

    await runner.run(make_turn(prompts=(prompt,)), discard())

    # NOTE: segments[0] is the current-time stamp every user message carries
    assert llm.requests[0].messages[-1].segments[1:] == (
        TextSegment(text='remind me at six'),
    )


async def test_an_attachment_the_model_cannot_take_arrives_as_its_hint() -> None:
    llm = answering()
    runner = AgentRunner(llm, make_toolset(), ClockManager())
    voice = attachment('audio', mime='audio/ogg', hint='[audio: audio/ogg 12s]')
    prompt = TurnPrompt(text='', actor='user:1', attachments=(voice,))

    await runner.run(make_turn(prompts=(prompt,)), discard())

    # NOTE: segments[0] is the current-time stamp every user message carries
    assert llm.requests[0].messages[-1].segments[1:] == (
        TextSegment(text='[audio: audio/ogg 12s]'),
    )


async def test_a_prompt_with_nothing_in_it_is_not_a_message() -> None:
    llm = answering()
    runner = AgentRunner(llm, make_toolset(), ClockManager())
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
    tools = ToolSet()
    tools.add('probe', probe)
    runner = AgentRunner(llm, tools, ClockManager())

    with bound_contextvars(turn_id='opened-by-the-channel'):
        await runner.run(make_turn(), discard())

    assert probe.seen['turn_id'] == 'opened-by-the-channel'
