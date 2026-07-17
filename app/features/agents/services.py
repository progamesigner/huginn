import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import cast
from uuid import uuid4

from structlog.contextvars import bound_contextvars

from app.features.tools import ToolRegistry
from lib.llm import (
    CompletionChunk,
    CompletionRequest,
    FinishChunk,
    LLMClient,
    Message,
    Segment,
    TextChunk,
    TextSegment,
    ThinkingChunk,
    ToolCallChunk,
    ToolCallSegment,
    ToolResultSegment,
)

from .exceptions import TurnTimeout

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 10
_DEFAULT_TURN_TIMEOUT = 300.0
_DEFAULT_TOOL_TIMEOUT = 60.0


@dataclass(frozen=True)
class ToolExecution:
    name: str
    duration: float
    ok: bool


@dataclass(frozen=True)
class TurnInput:
    instructions: str
    user_input: str
    model: str
    prior_turns: tuple[Message, ...] = ()
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    turn_timeout: float = _DEFAULT_TURN_TIMEOUT
    tool_timeout: float = _DEFAULT_TOOL_TIMEOUT


@dataclass(frozen=True)
class TurnOutput:
    text: str
    completed: bool = True
    steps: tuple[Step, ...] = ()


@dataclass(frozen=True)
class Step:
    tools: tuple[ToolExecution, ...] = ()


class ToolCallBuilder:
    id: str
    name: str
    arguments: str

    def __init__(self) -> None:
        self.id = ''
        self.name = ''
        self.arguments = ''


def build_messages(turn: TurnInput) -> Iterator[Message]:
    yield Message(role='system', segments=(TextSegment(text=turn.instructions),))
    yield from turn.prior_turns
    yield Message(role='user', segments=(TextSegment(text=turn.user_input),))


def assistant_message(text: str, tool_calls: tuple[ToolCallSegment, ...]) -> Message:
    segments: tuple[Segment, ...] = (
        (TextSegment(text=text), *tool_calls) if text else tool_calls
    )
    return Message(role='assistant', segments=segments)


def tool_result_messages(results: tuple[ToolResultSegment, ...]) -> Iterator[Message]:
    return (Message(role='tool', segments=(result,)) for result in results)


async def execute(
    registry: ToolRegistry,
    call: ToolCallSegment,
    tool_timeout: float,
) -> tuple[ToolResultSegment, ToolExecution]:
    started = time.monotonic()
    try:
        async with asyncio.timeout(tool_timeout):
            result = await registry.execute(call)
    except TimeoutError:
        logger.warning('Tool %s timed out after %.1fs', call.name, tool_timeout)
        result = ToolResultSegment(
            tool_call_id=call.id,
            content=f'tool {call.name!r} timed out',
            error=True,
        )

    record = ToolExecution(
        name=call.name,
        duration=time.monotonic() - started,
        ok=not result.error,
    )
    logger.info(
        'Tool %s finished in %.3fs (ok=%s)',
        record.name,
        record.duration,
        record.ok,
    )
    return result, record


async def execute_calls(
    registry: ToolRegistry,
    calls: tuple[ToolCallSegment, ...],
    tool_timeout: float,
) -> tuple[tuple[ToolResultSegment, ...], Step]:
    tasks = (execute(registry, call, tool_timeout) for call in calls)
    results, records = cast(
        tuple[tuple[ToolResultSegment, ...], tuple[ToolExecution, ...]],
        zip(*await asyncio.gather(*tasks), strict=True),
    )
    return results, Step(tools=records)


async def consume(
    stream: AsyncIterator[CompletionChunk],
    on_text: Callable[[str], None],
) -> tuple[str, tuple[ToolCallSegment, ...]]:
    fragments = []
    calls = cast(dict[int, ToolCallBuilder], {})

    async for chunk in stream:
        match chunk:
            case TextChunk(text=fragment):
                fragments.append(fragment)
                on_text(fragment)
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
    registry: ToolRegistry

    def __init__(self, llm: LLMClient, registry: ToolRegistry) -> None:
        self.llm = llm
        self.registry = registry

    async def run(self, turn: TurnInput, on_text: Callable[[str], None]) -> TurnOutput:
        with bound_contextvars(turn_id=uuid4().hex):
            return await self._process(turn, on_text)

    async def _process(
        self,
        turn: TurnInput,
        on_text: Callable[[str], None],
    ) -> TurnOutput:
        history = list(build_messages(turn))
        steps = []
        text = ''

        try:
            async with asyncio.timeout(turn.turn_timeout):
                for iteration in range(turn.max_iterations):
                    request = CompletionRequest(
                        model=turn.model,
                        messages=tuple(history),
                        tools=self.registry.schemas(),
                    )
                    text, tool_calls = await consume(self.llm.stream(request), on_text)

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
                    results, step = await execute_calls(
                        self.registry,
                        tool_calls,
                        turn.tool_timeout,
                    )
                    steps.append(step)
                    history.append(assistant_message(text, tool_calls))
                    history.extend(tool_result_messages(results))
        except TimeoutError as error:
            logger.warning('Turn exceeded turn_timeout=%.1fs', turn.turn_timeout)
            raise TurnTimeout(
                f'turn exceeded turn_timeout={turn.turn_timeout}s'
            ) from error

        logger.warning(
            'Turn hit max_iterations=%d without finishing',
            turn.max_iterations,
        )
        return TurnOutput(text=text, completed=False, steps=tuple(steps))
