from typing import Annotated, override

from pydantic import Field

from app.core.tools import Tool, ToolOutput, ToolParams


class GreetParams(ToolParams):
    name: Annotated[str, Field()]


class GreetTool(Tool[GreetParams]):
    name = 'greet'
    description = 'Greet someone by name'
    params = GreetParams

    @override
    async def run(self, params: GreetParams) -> ToolOutput:
        return f'Hello, {params.name}!'


TOOLS = (GreetTool(),)
