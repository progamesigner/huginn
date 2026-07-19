from typing import Annotated, Any, cast, override

from pydantic import Field

from app.features.tools.binding import ConfiguredTool, build_bound_tool, check_overlay
from app.features.tools.config import ToolConfig
from app.features.tools.registry import ToolRegistry, ToolSet
from app.features.tools.types import Tool, ToolOutput, ToolParams, TypedTool


class SearchParams(ToolParams):
    query: Annotated[str, Field(description='what to search')]
    max_results: Annotated[int, Field(ge=1, le=10)] = 5


class SearchTool(TypedTool[SearchParams]):
    name = 'search'
    description = 'Search the web'
    params = SearchParams

    calls: list[SearchParams]

    def __init__(self) -> None:
        self.calls = []

    @override
    async def run(self, params: SearchParams) -> ToolOutput:
        self.calls.append(params)
        return f'{params.query}:{params.max_results}'


def _schema(tool: Tool) -> dict[str, Any]:
    schema = tool.schema()
    assert isinstance(schema, dict)
    return schema


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(SearchTool)
    return registry


def test_schema_writes_param_default_and_drops_key_from_required() -> None:
    configured = ConfiguredTool(SearchTool(), {'query': 'weather'}, {}, 'desc')

    schema = _schema(configured)

    assert schema['properties']['query']['default'] == 'weather'
    assert 'query' not in schema.get('required', [])


def test_schema_overrides_existing_default_for_optional_param() -> None:
    configured = ConfiguredTool(SearchTool(), {'max_results': 3}, {}, 'desc')

    schema = _schema(configured)

    assert schema['properties']['max_results']['default'] == 3


def test_schema_hides_injected_keys_entirely() -> None:
    configured = ConfiguredTool(SearchTool(), {}, {'max_results': 7}, 'desc')

    schema = _schema(configured)

    assert 'max_results' not in schema['properties']
    assert 'max_results' not in schema.get('required', [])


async def test_invoke_prefers_inject_over_model_over_params() -> None:
    tool = SearchTool()
    configured = ConfiguredTool(tool, {'max_results': 3}, {'query': 'locked'}, 'desc')

    await configured.invoke('{"query": "from-model", "max_results": 9}')

    assert tool.calls[-1].query == 'locked'
    assert tool.calls[-1].max_results == 9


async def test_invoke_falls_back_to_param_default_when_model_omits_key() -> None:
    tool = SearchTool()
    configured = ConfiguredTool(tool, {'max_results': 3}, {}, 'desc')

    await configured.invoke('{"query": "hi"}')

    assert tool.calls[-1].max_results == 3


async def test_invoke_accepts_empty_arguments_when_overlay_supplies_all() -> None:
    tool = SearchTool()
    configured = ConfiguredTool(tool, {'query': 'q', 'max_results': 2}, {}, 'desc')

    await configured.invoke('')

    assert tool.calls[-1].query == 'q'
    assert tool.calls[-1].max_results == 2


def test_build_bound_tool_uses_class_description_by_default() -> None:
    tool = build_bound_tool(_registry(), ToolConfig(kind='search'), {}, {})

    assert tool.description == 'Search the web'


def test_build_bound_tool_applies_description_override() -> None:
    config = ToolConfig(kind='search', description='Premium search, wide access')

    tool = build_bound_tool(_registry(), config, {}, {})

    assert tool.description == 'Premium search, wide access'


def test_same_kind_instances_are_distinguishable_in_a_tool_set() -> None:
    registry = _registry()
    tools = ToolSet()
    tools.add(
        'search',
        build_bound_tool(
            registry, ToolConfig(kind='search', description='Basic'), {}, {}
        ),
    )
    tools.add(
        'supersearch',
        build_bound_tool(
            registry, ToolConfig(kind='search', description='Deep'), {}, {}
        ),
    )

    schemas = cast(tuple[dict[str, Any], ...], tools.schemas())
    named = {
        entry['function']['name']: entry['function']['description'] for entry in schemas
    }

    assert named == {'search': 'Basic', 'supersearch': 'Deep'}


def test_check_overlay_accepts_valid_params_and_inject() -> None:
    assert (
        check_overlay(_registry(), 'search', {'max_results': 3}, {'query': 'x'}) is None
    )


def test_check_overlay_rejects_unknown_param_key() -> None:
    problem = check_overlay(_registry(), 'search', {'bogus': 1}, {})

    assert problem is not None
    assert 'bogus' in problem


def test_check_overlay_rejects_unknown_inject_key() -> None:
    problem = check_overlay(_registry(), 'search', {}, {'bogus': 1})

    assert problem is not None
    assert 'bogus' in problem


def test_check_overlay_rejects_overlapping_keys() -> None:
    problem = check_overlay(_registry(), 'search', {'query': 'a'}, {'query': 'b'})

    assert problem is not None
    assert 'query' in problem


def test_check_overlay_rejects_wrong_value_type() -> None:
    problem = check_overlay(_registry(), 'search', {'max_results': 'lots'}, {})

    assert problem is not None
    assert 'max_results' in problem


def test_check_overlay_rejects_constraint_violation() -> None:
    problem = check_overlay(_registry(), 'search', {'max_results': 99}, {})

    assert problem is not None
    assert 'max_results' in problem
