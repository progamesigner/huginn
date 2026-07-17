from typing import Annotated, override

import pytest
from pydantic import BaseModel, Field

from app.features.tools import (
    DuplicateToolError,
    Tool,
    ToolOutput,
    ToolParams,
    ToolRegistry,
    ToolSet,
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


class GreetConfig(BaseModel):
    greeting: str


def test_register_declares_tool_by_kind() -> None:
    registry = ToolRegistry()

    registry.register(GreetTool)

    assert registry.get('greet') is GreetTool


def test_register_rejects_duplicate_kind() -> None:
    registry = ToolRegistry()
    registry.register(GreetTool)

    with pytest.raises(DuplicateToolError, match='greet'):
        registry.register(GreetTool)


def test_get_reports_unknown_kind_with_available_list() -> None:
    registry = ToolRegistry()
    registry.register(GreetTool)

    with pytest.raises(UnknownToolError, match='greet'):
        registry.get('missing')


def test_build_constructs_through_registered_builder() -> None:
    built: list[GreetConfig] = []

    def build(config: BaseModel) -> GreetTool:
        assert isinstance(config, GreetConfig)
        built.append(config)
        return GreetTool()

    registry = ToolRegistry()
    registry.register(GreetTool, build=build)

    tool = registry.build('greet', GreetConfig(greeting='hi'))

    assert isinstance(tool, GreetTool)
    assert built == [GreetConfig(greeting='hi')]


def test_build_defaults_to_argless_construction() -> None:
    registry = ToolRegistry()
    registry.register(GreetTool)

    assert isinstance(registry.build('greet', GreetConfig(greeting='hi')), GreetTool)


def test_build_rejects_unknown_kind() -> None:
    registry = ToolRegistry()

    with pytest.raises(UnknownToolError, match='none'):
        registry.build('greet', GreetConfig(greeting='hi'))


def test_toolset_schemas_use_instance_names() -> None:
    tools = ToolSet()
    tools.add('greet-a', GreetTool())
    tools.add('greet-b', GreetTool())

    assert tools.schemas() == (
        {
            'type': 'function',
            'function': {
                'name': 'greet-a',
                'description': 'Greet someone by name',
                'parameters': GreetParams.model_json_schema(),
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'greet-b',
                'description': 'Greet someone by name',
                'parameters': GreetParams.model_json_schema(),
            },
        },
    )
