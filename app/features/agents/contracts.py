from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.contracts.channels import ChannelReplyContract, ToolPhase
from lib.llm import Message, ModelRef

from .config import AgentConfig
from .types import Tool, TurnPrompt, TurnRequest

type NamedTool = tuple[str, Tool]
type ToolSourceFactory = Callable[[], ToolSource]


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
    async def collect(self, agent: AgentConfig) -> ToolHarvest: ...


class TranscriptStoreContract(ABC):
    @abstractmethod
    async def history(
        self,
        channel: str,
        chat: str,
        agent: str,
    ) -> tuple[Message, ...]: ...

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
    factories: list[ToolSourceFactory]

    def __init__(self) -> None:
        self.factories = []

    def register(self, factory: ToolSourceFactory) -> None:
        self.factories.append(factory)

    def sources(self) -> tuple[ToolSource, ...]:
        return tuple(factory() for factory in self.factories)
