import asyncio
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import cast
from uuid import uuid4

from structlog.contextvars import bound_contextvars, get_contextvars

from lib.llm import (
    CompletionChunk,
    CompletionRequest,
    FinishChunk,
    LLMClient,
    Message,
    TextChunk,
    TextSegment,
    ThinkingChunk,
    ToolCallChunk,
    ToolCallSegment,
    ToolResultSegment,
)
from lib.temporal import ClockContract

from .config import AgentConfig
from .contracts import TextSink, ToolSource
from .exceptions import AgentError, TurnTimeoutError
from .types import Step, ToolExecution, ToolSet, TurnInput, TurnOutput

logger = logging.getLogger(__name__)


class ToolCallBuilder:
    id: str
    name: str
    arguments: str

    def __init__(self) -> None:
        self.id = ''
        self.name = ''
        self.arguments = ''


def _build_messages(turn: TurnInput) -> Iterator[Message]:
    yield Message(role='system', segments=(TextSegment(text=turn.instructions),))
    yield from turn.prior_turns
    for prompt in turn.prompts:
        yield Message(role='user', segments=(TextSegment(text=prompt),))


def _assistant_message(text: str, tool_calls: tuple[ToolCallSegment, ...]) -> Message:
    segments = (TextSegment(text=text), *tool_calls) if text else tool_calls
    return Message(role='assistant', segments=segments)


def _tool_result_messages(results: tuple[ToolResultSegment, ...]) -> Iterator[Message]:
    return (Message(role='tool', segments=(result,)) for result in results)


def _turn_id() -> str:
    # NOTE: whoever opens the turn owns its id -- a channel binds one before the reply
    # boundary it needs the id for, so minting here would give that turn a second name
    bound = get_contextvars().get('turn_id')
    return bound if isinstance(bound, str) else uuid4().hex


async def _execute(
    tools: ToolSet,
    call: ToolCallSegment,
    tool_timeout: float,
    clock: ClockContract,
) -> tuple[ToolResultSegment, ToolExecution]:
    timer = clock.timer()
    try:
        async with asyncio.timeout(tool_timeout):
            result = await tools.execute(call)
    except TimeoutError:
        logger.warning('Tool %s timed out after %.1fs', call.name, tool_timeout)
        result = ToolResultSegment(
            tool_call_id=call.id,
            content=f'tool {call.name!r} timed out',
            error=True,
        )

    record = ToolExecution(
        name=call.name,
        duration=timer.elapsed().total_seconds(),
        ok=not result.error,
    )
    logger.info(
        'Tool %s finished in %.3fs (ok=%s)',
        record.name,
        record.duration,
        record.ok,
    )
    return result, record


async def _execute_calls(
    tools: ToolSet,
    calls: tuple[ToolCallSegment, ...],
    tool_timeout: float,
    clock: ClockContract,
) -> tuple[tuple[ToolResultSegment, ...], Step]:
    tasks = (_execute(tools, call, tool_timeout, clock) for call in calls)
    results, records = cast(
        tuple[tuple[ToolResultSegment, ...], tuple[ToolExecution, ...]],
        zip(*await asyncio.gather(*tasks), strict=True),
    )
    return results, Step(tools=records)


async def _consume(
    stream: AsyncIterator[CompletionChunk],
    on_text: TextSink,
) -> tuple[str, tuple[ToolCallSegment, ...]]:
    fragments = []
    calls = {}

    async for chunk in stream:
        match chunk:
            case TextChunk(text=fragment):
                fragments.append(fragment)
                await on_text(fragment)
            case ToolCallChunk(index=index, id=id_, name=name, arguments=arguments):
                builder = calls.setdefault(index, ToolCallBuilder())
                if id_ is not None:
                    builder.id = id_
                if name is not None:
                    builder.name = name
                builder.arguments += arguments
            case ThinkingChunk():
                pass
            case FinishChunk():
                pass

    tool_calls = tuple(
        ToolCallSegment(id=builder.id, name=builder.name, arguments=builder.arguments)
        for _, builder in sorted(calls.items())
    )
    return ''.join(fragments), tool_calls


class AgentRunner:
    llm: LLMClient
    tools: ToolSet
    clock: ClockContract

    def __init__(self, llm: LLMClient, tools: ToolSet, clock: ClockContract) -> None:
        self.llm = llm
        self.tools = tools
        self.clock = clock

    async def run(self, turn: TurnInput, on_text: TextSink) -> TurnOutput:
        with bound_contextvars(turn_id=_turn_id()):
            return await self._process(turn, on_text)

    async def _process(self, turn: TurnInput, on_text: TextSink) -> TurnOutput:
        history = list(_build_messages(turn))
        steps = []
        text = ''

        try:
            async with asyncio.timeout(turn.turn_timeout):
                for iteration in range(turn.max_iterations):
                    request = CompletionRequest(
                        model=turn.model,
                        messages=tuple(history),
                        tools=self.tools.schemas(),
                    )
                    text, tool_calls = await _consume(self.llm.stream(request), on_text)

                    if not tool_calls:
                        logger.info(
                            'Turn completed after %d iterations',
                            iteration + 1,
                        )
                        return TurnOutput(text=text, steps=(*steps, Step()))

                    logger.info(
                        'Iteration %d executing %d tool calls',
                        iteration,
                        len(tool_calls),
                    )
                    results, step = await _execute_calls(
                        self.tools,
                        tool_calls,
                        turn.tool_timeout,
                        self.clock,
                    )
                    steps.append(step)
                    history.append(_assistant_message(text, tool_calls))
                    history.extend(_tool_result_messages(results))
        except TimeoutError as error:
            logger.warning('Turn exceeded turn_timeout=%.1fs', turn.turn_timeout)
            raise TurnTimeoutError(
                f'turn exceeded turn_timeout={turn.turn_timeout}s',
            ) from error

        logger.warning(
            'Turn hit max_iterations=%d without finishing',
            turn.max_iterations,
        )
        return TurnOutput(text=text, completed=False, steps=tuple(steps))


async def build_tool_set(
    sources: Sequence[ToolSource],
    agent: AgentConfig,
) -> ToolSet:
    tools = ToolSet()
    for source in sources:
        for name, tool in await source.collect(agent):
            tools.add(name, tool)

    return tools


def load_instructions(agent: AgentConfig) -> str:
    if agent.instructions is not None:
        return agent.instructions

    if agent.instructions_file is not None:
        return Path(agent.instructions_file).read_text(encoding='utf-8')

    raise AgentError('agent has neither instructions nor instructions_file')
