from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from app.contracts.channels import ChannelReplyContract, ToolPhase
from lib.llm import Message, ModelRef

from .config import AgentConfig
from .entities import (
    RoutingHistory,
    RoutingTurn,
    Tool,
    TurnPrompt,
    TurnRequest,
    TurnScope,
)

type NamedTool = tuple[str, Tool]
type ToolSourceCollector = Callable[[], Iterable[ToolSource]]


class TurnStream(Protocol):
    async def on_routed(self, agent: str) -> None: ...

    async def on_text(self, fragment: str) -> None: ...

    async def on_tool(self, name: str, phase: ToolPhase) -> None: ...


@dataclass(frozen=True)
class ToolHarvest:
    tools: tuple[NamedTool, ...] = ()
    unreachable: tuple[str, ...] = ()


class ToolSource(ABC):
    @abstractmethod
    async def collect(self, agent: AgentConfig, scope: TurnScope) -> ToolHarvest: ...


class TranscriptStoreContract(ABC):
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
    ) -> None: ...

    @abstractmethod
    async def maintain(
        self,
        channel: str,
        chat: str,
        agent: str,
        ref: ModelRef,
    ) -> None: ...


class TurnHandlerContract(ABC):
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
