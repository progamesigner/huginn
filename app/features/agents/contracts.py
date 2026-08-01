from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from app.contracts.channels import ChannelReplyContract, ToolPhase
from lib.llm import Message

from .config import AgentConfig
from .types import Tool, TurnPrompt, TurnRequest

type NamedTool = tuple[str, Tool]
type ToolSourceFactory = Callable[[], ToolSource]


class TurnStream(Protocol):
    async def on_routed(self, agent: str) -> None: ...

    async def on_text(self, fragment: str) -> None: ...

    async def on_tool(self, name: str, phase: ToolPhase) -> None: ...


class ToolSource(ABC):
    @abstractmethod
    async def collect(self, agent: AgentConfig) -> Sequence[NamedTool]: ...


class TranscriptContract(ABC):
    @abstractmethod
    async def history(self) -> tuple[Message, ...]: ...

    @abstractmethod
    async def record(self, prompts: Sequence[TurnPrompt], answer: str) -> None: ...


class TranscriptStoreContract(ABC):
    @abstractmethod
    def open(
        self,
        channel: str,
        chat: str,
        agent: str,
    ) -> AbstractAsyncContextManager[TranscriptContract]: ...

    @abstractmethod
    async def maintain(
        self,
        channel: str,
        chat: str,
        agent: str,
        model: str,
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
