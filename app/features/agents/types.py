import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import override

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from app.contracts.channels import AttachmentKind, TurnEnding
from lib.llm import Message, ToolCallSegment, ToolResultSegment

from .config import AgentConfig
from .exceptions import DuplicateToolNameError

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 10
_DEFAULT_TURN_TIMEOUT = 300.0
_DEFAULT_TOOL_TIMEOUT = 60.0
_DEFAULT_MIN_ITERATION_BUDGET = 15.0

type Instances = dict[str, Tool]
type ToolOutput = str


@dataclass(frozen=True)
class SelectionPolicy:
    fallback: str
    model: str | None = None
    provider: str | None = None


@dataclass(frozen=True)
class RoutingPolicy:
    bindings: Mapping[str, str] = field(default_factory=dict)
    agent: str | None = None
    selection: SelectionPolicy | None = None
    allowed: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RoutedAgent:
    name: str
    config: AgentConfig


@dataclass(frozen=True)
class PromptAttachment:
    kind: AttachmentKind
    mime: str
    reference: str
    unique: str
    data: bytes
    hint: str
    transcript: str | None = None


@dataclass(frozen=True)
class TurnPrompt:
    text: str
    actor: str
    attachments: tuple[PromptAttachment, ...] = ()


@dataclass(frozen=True)
class TurnRequest:
    channel: str
    chat: str
    policy: RoutingPolicy
    prompts: tuple[TurnPrompt, ...]
    denied_tools: frozenset[str] = field(default_factory=frozenset)
    max_completion_tokens: int | None = None


@dataclass(frozen=True)
class ToolExecution:
    name: str
    duration: float
    ok: bool


@dataclass(frozen=True)
class Step:
    tools: tuple[ToolExecution, ...] = ()


@dataclass(frozen=True)
class TurnBudget:
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    turn_timeout: float = _DEFAULT_TURN_TIMEOUT
    tool_timeout: float = _DEFAULT_TOOL_TIMEOUT
    min_iteration_budget: float = _DEFAULT_MIN_ITERATION_BUDGET


def resolve_budget(agent: AgentConfig) -> TurnBudget:
    default = TurnBudget()
    return TurnBudget(
        max_iterations=agent.max_iterations or default.max_iterations,
        turn_timeout=agent.turn_timeout or default.turn_timeout,
        tool_timeout=agent.tool_timeout or default.tool_timeout,
    )


@dataclass(frozen=True)
class TurnInput:
    instructions: str
    prompts: tuple[TurnPrompt, ...]
    model: str
    now: datetime
    channel: str
    prior_turns: tuple[Message, ...] = ()
    max_completion_tokens: int | None = None
    budget: TurnBudget = field(default_factory=TurnBudget)


@dataclass(frozen=True)
class TurnOutput:
    text: str
    ending: TurnEnding = 'complete'
    steps: tuple[Step, ...] = ()

    @property
    def completed(self) -> bool:
        return self.ending == 'complete'


class ToolParams(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    async def invoke(self, arguments: str) -> ToolOutput: ...

    @abstractmethod
    def schema(self) -> JsonValue: ...


class TypedTool[P: ToolParams](Tool):
    params: type[P]

    @abstractmethod
    async def run(self, params: P) -> ToolOutput: ...

    @override
    async def invoke(self, arguments: str) -> ToolOutput:
        return await self.run(self.params.model_validate_json(arguments))

    @override
    def schema(self) -> JsonValue:
        return self.params.model_json_schema()


class ToolSet:
    tools: Instances

    def __init__(self, tools: Mapping[str, Tool] | None = None) -> None:
        self.tools = dict(tools or {})

    def add(self, name: str, tool: Tool) -> None:
        if name in self.tools:
            raise DuplicateToolNameError(f'tool {name!r} is already in the set')

        self.tools[name] = tool

    def schemas(self) -> tuple[JsonValue, ...]:
        return tuple(
            {
                'type': 'function',
                'function': {
                    'name': name,
                    'description': tool.description,
                    'parameters': tool.schema(),
                },
            }
            for name, tool in self.tools.items()
        )

    async def execute(self, call: ToolCallSegment) -> ToolResultSegment:
        if call.name not in self.tools:
            available = ', '.join(sorted(self.tools)) or 'none'
            return ToolResultSegment(
                tool_call_id=call.id,
                content=f'unknown tool {call.name!r}; available: {available}',
                error=True,
            )

        tool = self.tools[call.name]
        try:
            output = await tool.invoke(call.arguments)
        except ValidationError as error:
            return ToolResultSegment(
                tool_call_id=call.id,
                content=str(error),
                error=True,
            )
        except Exception as error:
            logger.exception('Tool %s raised while executing', call.name)
            return ToolResultSegment(
                tool_call_id=call.id,
                content=f'tool {call.name!r} failed: {error}',
                error=True,
            )

        return ToolResultSegment(tool_call_id=call.id, content=output, error=False)
