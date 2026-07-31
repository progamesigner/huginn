import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import override

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from app.contracts.channels import AttachmentKind
from lib.llm import Message, ToolCallSegment, ToolResultSegment

from .config import AgentConfig
from .exceptions import DuplicateToolNameError

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 10
_DEFAULT_TURN_TIMEOUT = 300.0
_DEFAULT_TOOL_TIMEOUT = 60.0

type Instances = dict[str, Tool]
type ToolOutput = str


@dataclass(frozen=True)
class SelectionPolicy:
    fallback: str
    model: str | None = None


@dataclass(frozen=True)
class RoutingPolicy:
    bindings: Mapping[str, str] = field(default_factory=dict)
    agent: str | None = None
    selection: SelectionPolicy | None = None


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


@dataclass(frozen=True)
class ToolExecution:
    name: str
    duration: float
    ok: bool


@dataclass(frozen=True)
class Step:
    tools: tuple[ToolExecution, ...] = ()


@dataclass(frozen=True)
class TurnInput:
    instructions: str
    prompts: tuple[TurnPrompt, ...]
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
        except Exception:
            logger.exception('Tool %s raised while executing', call.name)
            return ToolResultSegment(
                tool_call_id=call.id,
                content=f'tool {call.name!r} failed',
                error=True,
            )

        return ToolResultSegment(tool_call_id=call.id, content=output, error=False)
