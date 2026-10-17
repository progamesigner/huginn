import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, override

from pydantic import BaseModel, ConfigDict, JsonValue

from app.contracts.channels import ChannelReplyContract, ReplyArtifact, ToolPhase
from lib.llm import Message, ModelRef

from .config import AgentConfig
from .entities import (
    RoutingHistory,
    RoutingPolicy,
    RoutingTurn,
    TurnBudget,
    TurnOutput,
    TurnPrompt,
    TurnRequest,
    TurnScope,
)

type NamedTool = tuple[str, Tool]
type ToolOutput = str | ToolResult
type ToolSourceCollector = Callable[[], Iterable[ToolSource]]


# NOTE: the model reads `text` alone; the artifacts go to the reader beside it
@dataclass(frozen=True)
class ToolResult:
    text: str
    artifacts: tuple[ReplyArtifact, ...] = ()


class TurnStream(Protocol):
    async def on_artifact(self, artifact: ReplyArtifact) -> None: ...

    async def on_routed(self, agent: str) -> None: ...

    async def on_text(self, fragment: str) -> None: ...

    async def on_tool(self, name: str, phase: ToolPhase) -> None: ...


class ToolParams(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Tool(ABC):
    name: str
    description: str
    execution_policy: Literal['ordinary', 'delegation'] = 'ordinary'
    writes_brief: bool = False
    inherits_to_child: bool = True

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


@dataclass(frozen=True)
class ToolHarvest:
    tools: tuple[NamedTool, ...] = ()
    unreachable: tuple[str, ...] = ()
    context: tuple[str, ...] = ()


class ToolSource(ABC):
    @abstractmethod
    async def collect(self, agent: AgentConfig, scope: TurnScope) -> ToolHarvest: ...


class TranscriptServiceContract(ABC):
    def routing_samples(
        self,
        channel: str,
        since: datetime,
        history_turns: int,
    ) -> AsyncIterator[RoutingHistory]:
        raise NotImplementedError('this transcript store cannot export routing samples')

    async def recent_turns(
        self,
        channel: str,
        chat: str,
        limit: int,
    ) -> tuple[tuple[RoutingTurn, ...], str | None]:
        return (), None

    @abstractmethod
    async def history(
        self,
        channel: str,
        chat: str,
    ) -> tuple[Message, ...]: ...

    @abstractmethod
    async def document_blobs(self, channel: str, chat: str) -> frozenset[str]: ...

    @abstractmethod
    async def record(
        self,
        channel: str,
        chat: str,
        agent: str,
        prompts: Sequence[TurnPrompt],
        answer: str,
        artifacts: Sequence[ReplyArtifact] = (),
    ) -> None: ...

    @abstractmethod
    async def maintain(
        self,
        channel: str,
        chat: str,
        agent: str,
        ref: ModelRef,
    ) -> None: ...


class TurnServiceContract(ABC):
    @abstractmethod
    async def handle(
        self,
        request: TurnRequest,
        reply: ChannelReplyContract,
    ) -> None: ...


class ToolSourceRegistry:
    collect: ToolSourceCollector

    # NOTE: sources are collected per turn, so each one sees the config of the moment
    def __init__(self, collect: ToolSourceCollector = tuple) -> None:
        self.collect = collect

    def sources(self) -> tuple[ToolSource, ...]:
        return tuple(self.collect())


@dataclass(frozen=True)
class HookHarvest:
    content: str | None = None
    unreachable: tuple[str, ...] = ()


class HookSource(ABC):
    @abstractmethod
    async def collect(self, agent: AgentConfig, scope: TurnScope) -> HookHarvest: ...


class HookRegistry:
    def __init__(self, collect: Callable[[], Iterable[HookSource]] = tuple) -> None:
        self.collect = collect

    def sources(self) -> tuple[HookSource, ...]:
        return tuple(self.collect())


@dataclass
class ExecutionContext:
    deadline: float
    semaphore: asyncio.Semaphore
    scope: TurnScope
    model: ModelRef
    budget: TurnBudget
    max_completion_tokens: int | None
    tools: dict[str, Tool]
    depth: int = 0
    user_messages: tuple[TurnPrompt, ...] = ()


@dataclass(frozen=True)
class SubagentDefinition:
    name: str
    source: str
    instructions: str
    model: str | None = None
    tools: tuple[str, ...] | None = None
    max_iterations: int | None = None
    max_completion_tokens: int | None = None
    turn_timeout: float | None = None


class SubagentStream(TurnStream):
    # Children stay silent; only their files reach the parent turn.
    def __init__(self) -> None:
        self.artifacts: list[ReplyArtifact] = []

    @override
    async def on_artifact(self, artifact: ReplyArtifact) -> None:
        self.artifacts.append(artifact)

    @override
    async def on_routed(self, agent: str) -> None:
        pass

    @override
    async def on_text(self, fragment: str) -> None:
        pass

    @override
    async def on_tool(self, name: str, phase: ToolPhase) -> None:
        pass


class SubagentServiceContract(ABC):
    @abstractmethod
    async def execute(
        self,
        definition: SubagentDefinition,
        task: str,
        context: ExecutionContext,
        stream: TurnStream | None = None,
    ) -> TurnOutput: ...


@dataclass(frozen=True)
class CommandInfo:
    name: str
    description: str


class ScopedCommandsServiceContract(ABC):
    @abstractmethod
    def enabled(self, policy: RoutingPolicy) -> bool: ...

    @abstractmethod
    async def inventory(self, request: TurnRequest) -> tuple[CommandInfo, ...]: ...

    @abstractmethod
    async def resolve(self, request: TurnRequest) -> TurnRequest: ...


class ExecutionAccess:
    # Runtime state is supplied by the runner, independently of persistent identity.
    @staticmethod
    def current() -> ExecutionContext | None:
        from .execution import current_execution

        return current_execution()
