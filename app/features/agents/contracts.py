from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence

from .config import AgentConfig
from .types import Tool

type NamedTool = tuple[str, Tool]
type TextSink = Callable[[str], Awaitable[None]]
type ToolSourceFactory = Callable[[], ToolSource]


class ToolSource(ABC):
    @abstractmethod
    async def collect(self, agent: AgentConfig) -> Sequence[NamedTool]: ...


class ToolSourceRegistry:
    factories: list[ToolSourceFactory]

    def __init__(self) -> None:
        self.factories = []

    def register(self, factory: ToolSourceFactory) -> None:
        self.factories.append(factory)

    def sources(self) -> tuple[ToolSource, ...]:
        return tuple(factory() for factory in self.factories)
