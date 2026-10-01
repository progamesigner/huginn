from typing import Annotated, override

import pytest
from pydantic import BaseModel, Field

from app.features.agents.entities import (
    ToolOutput,
    ToolParams,
    ToolSet,
    TurnScope,
    TypedTool,
)
from app.features.agents.exceptions import DuplicateToolNameError
from app.features.tools.exceptions import DuplicateToolError, UnknownToolError
from app.features.tools.registry import ToolKind, ToolRegistry

_SCOPE = TurnScope(actor='user:1', agent='scout', channel='support_bot', chat='1')


class GreetParams(ToolParams):
    name: Annotated[str, Field()]


class GreetTool(TypedTool[GreetParams]):
    name = 'greet'
    description = 'Greet someone by name'
    params = GreetParams

    @override
    async def run(self, params: GreetParams) -> ToolOutput:
        return f'Hello, {params.name}!'


class GreetConfig(BaseModel):
    greeting: str


def test_register_declares_tool_by_kind() -> None:
    registry = ToolRegistry([ToolKind(GreetTool)])

    assert registry.get('greet') is GreetTool


def test_register_rejects_duplicate_kind() -> None:
    with pytest.raises(DuplicateToolError, match='greet'):
        ToolRegistry([ToolKind(GreetTool), ToolKind(GreetTool)])


def test_get_reports_unknown_kind_with_available_list() -> None:
    registry = ToolRegistry(
        [
            ToolKind(GreetTool),
        ],
    )

    with pytest.raises(UnknownToolError, match='greet'):
        registry.get('missing')


def test_build_constructs_through_registered_builder() -> None:
    built = []

    def builder(config: BaseModel, scope: TurnScope) -> GreetTool:
        assert isinstance(config, GreetConfig)
        built.append(config)
        return GreetTool()

    registry = ToolRegistry(
        [
            ToolKind(GreetTool, builder=builder),
        ],
    )

    tool = registry.build('greet', GreetConfig(greeting='hi'), _SCOPE)

    assert isinstance(tool, GreetTool)
    assert built == [GreetConfig(greeting='hi')]


def test_build_defaults_to_argless_construction() -> None:
    registry = ToolRegistry(
        [
            ToolKind(GreetTool),
        ],
    )

    assert isinstance(
        registry.build('greet', GreetConfig(greeting='hi'), _SCOPE),
        GreetTool,
    )


def test_build_rejects_unknown_kind() -> None:
    registry = ToolRegistry()

    with pytest.raises(UnknownToolError, match='none'):
        registry.build('greet', GreetConfig(greeting='hi'), _SCOPE)


def test_register_records_config_model() -> None:
    registry = ToolRegistry(
        [
            ToolKind(GreetTool, GreetConfig),
        ],
    )

    assert registry.config_models == {'greet': GreetConfig}


def test_config_models_maps_configless_tools_to_none() -> None:
    registry = ToolRegistry(
        [
            ToolKind(GreetTool),
        ],
    )

    assert registry.config_models == {'greet': None}


def test_toolset_add_rejects_duplicate_name() -> None:
    tools = ToolSet()
    tools.add('greet', GreetTool())

    with pytest.raises(DuplicateToolNameError, match='greet'):
        tools.add('greet', GreetTool())


def test_params_model_returns_typed_tool_params() -> None:
    registry = ToolRegistry(
        [
            ToolKind(GreetTool),
        ],
    )

    assert registry.params_model('greet') is GreetParams


def test_params_model_returns_none_for_unknown_kind() -> None:
    assert ToolRegistry().params_model('missing') is None


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
