from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pydantic import BaseModel

from app.features.agents.types import Tool, TurnScope

from .exceptions import DuplicateToolError, UnknownToolError

type ToolBuilder = Callable[[BaseModel, TurnScope], Tool]
type Builders = dict[str, ToolBuilder]
type Configs = dict[str, type[BaseModel]]
type Tools = dict[str, type[Tool]]


@dataclass(frozen=True)
class ToolKind:
    tool: type[Tool]
    config: type[BaseModel] | None = None
    builder: ToolBuilder | None = None


class ToolRegistry:
    builders: Builders
    configs: Configs
    tools: Tools

    @property
    def config_models(self) -> dict[str, type[BaseModel] | None]:
        return {kind: self.configs.get(kind) for kind in self.tools}

    def __init__(self, kinds: Iterable[ToolKind] = ()) -> None:
        self.builders = {}
        self.configs = {}
        self.tools = {}
        for kind in kinds:
            self._add(kind)

    def _add(self, kind: ToolKind) -> None:
        tool = kind.tool
        if tool.name in self.tools:
            raise DuplicateToolError(f'tool {tool.name!r} is already registered')

        self.tools[tool.name] = tool
        self.builders[tool.name] = kind.builder or (lambda _config, _scope: tool())
        if kind.config is not None:
            self.configs[tool.name] = kind.config

    def build(self, kind: str, config: BaseModel, scope: TurnScope) -> Tool:
        self.get(kind)
        return self.builders[kind](config, scope)

    def get(self, kind: str) -> type[Tool]:
        if kind not in self.tools:
            available = ', '.join(sorted(self.tools)) or 'none'
            raise UnknownToolError(f'unknown tool {kind!r}; available: {available}')

        return self.tools[kind]

    def params_model(self, kind: str) -> type[BaseModel] | None:
        tool = self.tools.get(kind)
        return getattr(tool, 'params', None) if tool is not None else None
