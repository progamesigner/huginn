from collections.abc import Mapping

from pydantic import JsonValue, ValidationError

from app.contracts.channels import ReplyArtifact
from lib.llm import ToolCallSegment, ToolResultSegment
from lib.observability import Logger

from .contracts import Tool, ToolResult
from .exceptions import DuplicateToolNameError

type Instances = dict[str, Tool]
type ToolOutcome = tuple[ToolResultSegment, tuple[ReplyArtifact, ...]]


class ToolSet:
    logger: Logger
    tools: Instances

    def __init__(
        self,
        tools: Mapping[str, Tool] | None = None,
        *,
        logger: Logger,
    ) -> None:
        self.logger = logger.bind(component='tools')
        self.tools = dict(tools or {})
        self.dynamic_instructions: list[str] = []

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

    async def execute(self, call: ToolCallSegment) -> ToolOutcome:
        if call.name not in self.tools:
            available = ', '.join(sorted(self.tools)) or 'none'
            return _failed(
                call,
                f'unknown tool {call.name!r}; available: {available}',
            )

        tool = self.tools[call.name]
        try:
            output = await tool.invoke(call.arguments)
        except ValidationError as error:
            return _failed(call, str(error))
        except Exception as error:
            self.logger.exception('Tool %s raised while executing', call.name)
            return _failed(call, f'tool {call.name!r} failed: {error}')

        result = output if isinstance(output, ToolResult) else ToolResult(text=output)
        return (
            ToolResultSegment(tool_call_id=call.id, content=result.text, error=False),
            result.artifacts,
        )


def _failed(call: ToolCallSegment, content: str) -> ToolOutcome:
    return ToolResultSegment(tool_call_id=call.id, content=content, error=True), ()
