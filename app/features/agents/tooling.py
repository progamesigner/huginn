import logging
from collections.abc import Mapping

from pydantic import JsonValue, ValidationError

from lib.llm import ToolCallSegment, ToolResultSegment

from .contracts import Tool
from .exceptions import DuplicateToolNameError

logger = logging.getLogger(__name__)

type Instances = dict[str, Tool]


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
        except Exception as error:
            logger.exception('Tool %s raised while executing', call.name)
            return ToolResultSegment(
                tool_call_id=call.id,
                content=f'tool {call.name!r} failed: {error}',
                error=True,
            )

        return ToolResultSegment(tool_call_id=call.id, content=output, error=False)
