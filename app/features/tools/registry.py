import logging
from collections.abc import Callable, Mapping
from typing import cast

from pydantic import BaseModel, JsonValue, ValidationError

from lib.llm import ToolCallSegment, ToolResultSegment

from .exceptions import DuplicateToolError, UnknownToolError
from .types import Tool, ToolOutput, ToolParams

logger = logging.getLogger(__name__)

type ToolBuilder = Callable[[BaseModel], Tool[ToolParams]]
type Builders = dict[str, ToolBuilder]
type Configs = dict[str, type[BaseModel]]
type Instances = dict[str, Tool[ToolParams]]
type Tools = dict[str, type[Tool[ToolParams]]]


class ToolRegistry:
    builders: Builders
    configs: Configs
    tools: Tools

    @property
    def config_models(self) -> dict[str, type[BaseModel] | None]:
        return {kind: self.configs.get(kind) for kind in self.tools}

    def __init__(self) -> None:
        self.builders = cast(Builders, {})
        self.configs = cast(Configs, {})
        self.tools = cast(Tools, {})

    def register[P: ToolParams](
        self,
        tool: type[Tool[P]],
        config: type[BaseModel] | None = None,
        build: Callable[[BaseModel], Tool[P]] | None = None,
    ) -> None:
        if tool.name in self.tools:
            raise DuplicateToolError(f'tool {tool.name!r} is already registered')

        self.tools[tool.name] = cast(type[Tool[ToolParams]], tool)
        self.builders[tool.name] = cast(ToolBuilder, build or (lambda _: tool()))
        if config is not None:
            self.configs[tool.name] = config

    def build(self, kind: str, config: BaseModel) -> Tool[ToolParams]:
        self.get(kind)
        return self.builders[kind](config)

    def get(self, kind: str) -> type[Tool[ToolParams]]:
        if kind not in self.tools:
            available = ', '.join(sorted(self.tools)) or 'none'
            raise UnknownToolError(f'unknown tool {kind!r}; available: {available}')

        return self.tools[kind]


class ToolSet:
    tools: Instances

    def __init__(self, tools: Mapping[str, Tool[ToolParams]] | None = None) -> None:
        self.tools = dict(tools or {})

    def add[P: ToolParams](self, name: str, tool: Tool[P]) -> None:
        self.tools[name] = cast(Tool[ToolParams], tool)

    def schemas(self) -> tuple[JsonValue, ...]:
        return tuple(
            {
                'type': 'function',
                'function': {
                    'name': name,
                    'description': tool.description,
                    'parameters': tool.params.model_json_schema(),
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
            params = tool.params.model_validate_json(call.arguments)
        except ValidationError as error:
            return ToolResultSegment(
                tool_call_id=call.id,
                content=str(error),
                error=True,
            )

        try:
            output: ToolOutput = await tool.run(params)
        except Exception:
            logger.exception('Tool %s raised while executing', call.name)
            return ToolResultSegment(
                tool_call_id=call.id,
                content=f'tool {call.name!r} failed',
                error=True,
            )

        return ToolResultSegment(tool_call_id=call.id, content=output, error=False)
