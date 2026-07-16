from typing import Annotated, override

from pydantic import Field

from app.core.tools import Tool, ToolOutput, ToolParams


class EchoParams(ToolParams):
    text: Annotated[str, Field()]


class SecondEchoTool(Tool[EchoParams]):
    name = 'echo'
    description = 'Second echo tool'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        return params.text


TOOLS = (SecondEchoTool(),)
