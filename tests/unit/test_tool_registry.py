from typing import Annotated, override

import pytest
from pydantic import Field

from app.core.tools import (
    DuplicateToolError,
    Tool,
    ToolOutput,
    ToolParams,
    ToolRegistry,
    UnknownToolError,
)


class GreetParams(ToolParams):
    name: Annotated[str, Field()]


class GreetTool(Tool[GreetParams]):
    name = 'greet'
    description = 'Greet someone by name'
    params = GreetParams

    @override
    async def run(self, params: GreetParams) -> ToolOutput:
        return f'Hello, {params.name}!'


def test_register_adds_tool_by_name() -> None:
    registry = ToolRegistry()
    tool = GreetTool()

    registry.register(tool)

    assert registry.get('greet') is tool


def test_register_rejects_duplicate_name() -> None:
    registry = ToolRegistry()
    registry.register(GreetTool())

    with pytest.raises(DuplicateToolError, match='greet'):
        registry.register(GreetTool())


def test_get_reports_unknown_tool_with_available_list() -> None:
    registry = ToolRegistry()
    registry.register(GreetTool())

    with pytest.raises(UnknownToolError, match='greet'):
        registry.get('missing')


def test_schemas_produce_function_calling_shape() -> None:
    registry = ToolRegistry()
    registry.register(GreetTool())

    assert registry.schemas() == (
        {
            'type': 'function',
            'function': {
                'name': 'greet',
                'description': 'Greet someone by name',
                'parameters': GreetParams.model_json_schema(),
            },
        },
    )
