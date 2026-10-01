from typing import Annotated, cast, override

from pydantic import Field

from app.features.agents.config import BindingSource
from app.features.agents.contracts import ToolOutput, ToolParams, TypedTool
from app.features.tools.config import _check_bindings, _check_overlay
from app.features.tools.registry import ToolKind, ToolRegistry


class SearchParams(ToolParams):
    query: Annotated[str, Field(description='what to search')]
    max_results: Annotated[int, Field(ge=1, le=10)] = 5


class SearchTool(TypedTool[SearchParams]):
    name = 'search'
    description = 'Search the web'
    params = SearchParams

    @override
    async def run(self, params: SearchParams) -> ToolOutput:
        return f'{params.query}:{params.max_results}'


def _registry() -> ToolRegistry:
    registry = ToolRegistry(
        [
            ToolKind(SearchTool),
        ],
    )
    return registry


def test_check_overlay_accepts_valid_params_and_inject() -> None:
    assert (
        _check_overlay(_registry(), 'search', {'max_results': 3}, {'query': 'x'})
        is None
    )


def test_check_overlay_rejects_unknown_param_key() -> None:
    problem = _check_overlay(_registry(), 'search', {'bogus': 1}, {})

    assert problem is not None
    assert 'bogus' in problem


def test_check_overlay_rejects_unknown_inject_key() -> None:
    problem = _check_overlay(_registry(), 'search', {}, {'bogus': 1})

    assert problem is not None
    assert 'bogus' in problem


def test_check_overlay_rejects_overlapping_keys() -> None:
    problem = _check_overlay(_registry(), 'search', {'query': 'a'}, {'query': 'b'})

    assert problem is not None
    assert 'query' in problem


def test_check_overlay_rejects_wrong_value_type() -> None:
    problem = _check_overlay(_registry(), 'search', {'max_results': 'lots'}, {})

    assert problem is not None
    assert 'max_results' in problem


def test_check_overlay_rejects_constraint_violation() -> None:
    problem = _check_overlay(_registry(), 'search', {'max_results': 99}, {})

    assert problem is not None
    assert 'max_results' in problem


def test_check_bindings_accepts_known_parameter() -> None:
    assert (
        _check_bindings(
            _registry(),
            'search',
            {},
            cast(dict[str, BindingSource], {'query': 'actor'}),
        )
        is None
    )


def test_check_bindings_rejects_unknown_parameter() -> None:
    problem = _check_bindings(
        _registry(),
        'search',
        {},
        cast(dict[str, BindingSource], {'bogus': 'actor'}),
    )

    assert problem is not None
    assert 'bogus' in problem


def test_check_bindings_rejects_parameter_default_overlap() -> None:
    problem = _check_bindings(
        _registry(),
        'search',
        {'query': 'visible'},
        cast(dict[str, BindingSource], {'query': 'actor'}),
    )

    assert problem is not None
    assert 'query' in problem
