import httpx
import pytest
import respx
from app.features.tools.websearch import format_result
from pydantic import SecretStr, ValidationError

from app.exts.felo.client import FeloClient, SearchResource, SearchResult
from app.features.tools import WebSearchParams, WebSearchTool
from lib.http import HTTPClientManager

_PAYLOAD = {
    'status': 'ok',
    'data': {
        'answer': 'Taipei is sunny today.',
        'resources': [
            {
                'link': 'https://weather.example/taipei',
                'title': 'Taipei Weather',
                'snippet': 'Sunny, 31°C.',
            },
        ],
    },
}


def test_params_reject_unknown_fields_and_bad_bounds() -> None:
    with pytest.raises(ValidationError):
        WebSearchParams.model_validate({'query': 'x', 'unexpected': True})

    with pytest.raises(ValidationError):
        WebSearchParams.model_validate({'query': 'x', 'max_results': 0})

    with pytest.raises(ValidationError):
        WebSearchParams.model_validate({'query': 'x', 'max_results': 11})

    assert WebSearchParams.model_validate({'query': 'x'}).max_results == 5


def test_schema_describes_fields_for_llm() -> None:
    schema = WebSearchParams.model_json_schema()

    assert schema['required'] == ['query']
    assert schema['properties']['query']['description']
    assert schema['properties']['max_results']['description']


def test_format_result_lists_numbered_sources_after_answer() -> None:
    result = SearchResult(
        answer='An answer.',
        resources=(
            SearchResource(link='https://a.example', title='A', snippet='First.'),
            SearchResource(link='https://b.example', title='B', snippet=''),
        ),
    )

    assert format_result(result, max_results=5) == (
        'An answer.'
        '\n'
        '\nSources:'
        '\n1. A'
        '\n   https://a.example'
        '\n   First.'
        '\n2. B'
        '\n   https://b.example'
    )


def test_format_result_caps_sources_at_max_results() -> None:
    result = SearchResult(
        answer='',
        resources=tuple(
            SearchResource(link=f'https://example.com/{index}', title=f'T{index}')
            for index in range(10)
        ),
    )

    output = format_result(result, max_results=3)

    assert output.startswith('Sources:')
    assert '3. T2' in output
    assert 'T3' not in output


def test_format_result_shortens_long_snippets() -> None:
    result = SearchResult(
        resources=(
            SearchResource(
                link='https://example.com',
                title='Long',
                snippet='word ' * 100,
            ),
        ),
    )

    output = format_result(result, max_results=5)

    assert '[...]' in output
    assert all(len(line) <= 210 for line in output.splitlines())


def test_format_result_reports_when_empty() -> None:
    assert format_result(SearchResult(), max_results=5) == 'No results found.'


@respx.mock
async def test_run_searches_and_formats() -> None:
    respx.post('https://openapi.felo.ai/v2/chat').mock(
        return_value=httpx.Response(200, json=_PAYLOAD),
    )
    tool = WebSearchTool(
        FeloClient(
            http=HTTPClientManager(),
            api_key=SecretStr('felo-key'),
            base_url='https://openapi.felo.ai',
        )
    )

    output = await tool.run(WebSearchParams(query='taipei weather'))

    assert 'Taipei is sunny today.' in output
    assert '1. Taipei Weather' in output
    assert 'https://weather.example/taipei' in output
