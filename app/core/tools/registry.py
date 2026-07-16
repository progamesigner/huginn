import logging
from typing import cast

from pydantic import JsonValue, ValidationError

from lib.llm import ToolCallSegment, ToolResultSegment

from .base import Tool, ToolOutput, ToolParams
from .exceptions import DuplicateToolError, UnknownToolError

logger = logging.getLogger(__name__)

type Tools = dict[str, Tool[ToolParams]]


class ToolRegistry:
    tools: Tools

    def __init__(self) -> None:
        self.tools = cast(Tools, {})

    def register[P: ToolParams](self, tool: Tool[P]) -> None:
        if tool.name in self.tools:
            raise DuplicateToolError(f'tool {tool.name!r} is already registered')

        self.tools[tool.name] = cast(Tool[ToolParams], tool)

    def get(self, name: str) -> Tool[ToolParams]:
        if name not in self.tools:
            available = ', '.join(sorted(self.tools)) or 'none'
            raise UnknownToolError(f'unknown tool {name!r}; available: {available}')

        return self.tools[name]

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
            output: ToolOutput = await tool.run(params)
        except Exception:
            logger.exception('Tool %s raised while executing', call.name)
            return ToolResultSegment(
                tool_call_id=call.id,
                content=f'tool {call.name!r} failed',
                error=True,
            )

        return ToolResultSegment(tool_call_id=call.id, content=output, error=False)
