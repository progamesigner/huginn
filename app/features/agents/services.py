import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from app.core.tools import ToolRegistry
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

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 10


@dataclass(frozen=True)
class TurnInput:
    instructions: str
    user_input: str
    model: str
    prior_turns: tuple[Message, ...] = ()
    max_iterations: int = _DEFAULT_MAX_ITERATIONS


@dataclass(frozen=True)
class TurnOutput:
    text: str
    completed: bool = True


class ToolCallBuilder:
    id: str
    name: str
    arguments: str

    def __init__(self) -> None:
        self.id = ''
        self.name = ''
        self.arguments = ''


def build_messages(turn: TurnInput) -> list[Message]:
    return [
        Message(role='system', segments=(TextSegment(text=turn.instructions),)),
        *turn.prior_turns,
        Message(role='user', segments=(TextSegment(text=turn.user_input),)),
    ]


def assistant_message(text: str, tool_calls: tuple[ToolCallSegment, ...]) -> Message:
    segments: tuple[Segment, ...] = (
        (TextSegment(text=text), *tool_calls) if text else tool_calls
    )
    return Message(role='assistant', segments=segments)


def tool_result_messages(results: tuple[ToolResultSegment, ...]) -> tuple[Message, ...]:
    return tuple(Message(role='tool', segments=(result,)) for result in results)


async def execute_calls(
    registry: ToolRegistry,
    calls: tuple[ToolCallSegment, ...],
) -> tuple[ToolResultSegment, ...]:
    return tuple(await asyncio.gather(*(registry.execute(call) for call in calls)))


async def consume(
    stream: AsyncIterator[CompletionChunk],
    on_text: Callable[[str], None],
) -> tuple[str, tuple[ToolCallSegment, ...]]:
    fragments: list[str] = []
    calls: dict[int, ToolCallBuilder] = {}

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
        history = build_messages(turn)
        text = ''

        for _ in range(turn.max_iterations):
            request = CompletionRequest(
                model=turn.model,
                messages=tuple(history),
                tools=self.registry.schemas(),
            )
            text, tool_calls = await consume(self.llm.stream(request), on_text)

            if not tool_calls:
                return TurnOutput(text=text)

            results = await execute_calls(self.registry, tool_calls)
            history.append(assistant_message(text, tool_calls))
            history.extend(tool_result_messages(results))

        logger.warning(
            'Turn hit max_iterations=%d without finishing',
            turn.max_iterations,
        )
        return TurnOutput(text=text, completed=False)
