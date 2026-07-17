import logging
from collections.abc import Callable
from typing import cast

from pydantic import JsonValue, ValidationError

from lib.llm import ToolCallSegment, ToolResultSegment

from .exceptions import DuplicateToolError, UnknownToolError
from .types import Tool, ToolParams

logger = logging.getLogger(__name__)

type ToolFactory = Callable[[], Tool[ToolParams]]
type Factories = dict[str, ToolFactory]
type Instances = dict[str, Tool[ToolParams]]
type Tools = dict[str, type[Tool[ToolParams]]]


class ToolRegistry:
    factories: Factories
    instances: Instances
    tools: Tools

    def __init__(self) -> None:
        self.factories = cast(Factories, {})
        self.instances = cast(Instances, {})
        self.tools = cast(Tools, {})

    def register[P: ToolParams](
        self,
        tool: type[Tool[P]],
        factory: Callable[[], Tool[P]] | None = None,
    ) -> None:
        if tool.name in self.tools:
            raise DuplicateToolError(f'tool {tool.name!r} is already registered')

        self.tools[tool.name] = cast(type[Tool[ToolParams]], tool)
        self.factories[tool.name] = cast(ToolFactory, factory or tool)

    def get(self, name: str) -> type[Tool[ToolParams]]:
        if name not in self.tools:
            available = ', '.join(sorted(self.tools)) or 'none'
            raise UnknownToolError(f'unknown tool {name!r}; available: {available}')

        return self.tools[name]

    def resolve(self, name: str) -> Tool[ToolParams]:
        self.get(name)

        if name not in self.instances:
            self.instances[name] = self.factories[name]()

        return self.instances[name]

    def schemas(self) -> tuple[JsonValue, ...]:
        return tuple(
            {
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.params.model_json_schema(),
                },
            }
            for tool in self.tools.values()
        )

    async def execute(self, call: ToolCallSegment) -> ToolResultSegment:
        try:
            tool = self.get(call.name)
        except UnknownToolError as error:
            return ToolResultSegment(
                tool_call_id=call.id,
                content=str(error),
                error=True,
            )

        try:
            params = tool.params.model_validate_json(call.arguments)
        except ValidationError as error:
            return ToolResultSegment(
                tool_call_id=call.id,
                content=str(error),
                error=True,
            )

        try:
            output = await self.resolve(call.name).run(params)
        except Exception:
            logger.exception('Tool %s raised while executing', call.name)
            return ToolResultSegment(
                tool_call_id=call.id,
                content=f'tool {call.name!r} failed',
                error=True,
            )

        return ToolResultSegment(tool_call_id=call.id, content=output, error=False)
