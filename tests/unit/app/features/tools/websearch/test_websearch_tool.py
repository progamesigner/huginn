import httpx
import pytest
import respx
from pydantic import SecretStr, ValidationError

from app.features.tools.websearch.config import WebSearchToolConfig
from app.features.tools.websearch.repositories import (
    WebSearchRepository,
    WebSearchResult,
    WebSearchSource,
)
from app.features.tools.websearch.tool import (
    WebSearchParams,
    WebSearchTool,
    _format_result,
    build_websearch,
)
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
    result = WebSearchResult(
        answer='An answer.',
        sources=(
            WebSearchSource(link='https://a.example', title='A', snippet='First.'),
            WebSearchSource(link='https://b.example', title='B', snippet=''),
        ),
    )

    assert _format_result(result, max_results=5) == (
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
    result = WebSearchResult(
        answer='',
        sources=tuple(
            WebSearchSource(link=f'https://example.com/{index}', title=f'T{index}')
            for index in range(10)
        ),
    )

    output = _format_result(result, max_results=3)

    assert output.startswith('Sources:')
    assert '3. T2' in output
    assert 'T3' not in output


def test_format_result_shortens_long_snippets() -> None:
    result = WebSearchResult(
        sources=(
            WebSearchSource(
                link='https://example.com',
                title='Long',
                snippet='word ' * 100,
            ),
        ),
    )

    output = _format_result(result, max_results=5)

    assert '[...]' in output
    assert all(len(line) <= 210 for line in output.splitlines())


def test_format_result_reports_when_empty() -> None:
    assert _format_result(WebSearchResult(), max_results=5) == 'No results found.'


@respx.mock
async def test_run_searches_and_formats() -> None:
    respx.post('https://openapi.felo.ai/v2/chat').mock(
        return_value=httpx.Response(200, json=_PAYLOAD),
    )
    tool = WebSearchTool(
        WebSearchRepository(HTTPClientManager()),
        api_key=SecretStr('felo-key'),
        base_url='https://openapi.felo.ai',
    )

    output = await tool.run(WebSearchParams(query='taipei weather'))

    assert 'Taipei is sunny today.' in output
    assert '1. Taipei Weather' in output
    assert 'https://weather.example/taipei' in output


def test_build_websearch_binds_config_credentials() -> None:
    repository = WebSearchRepository(HTTPClientManager())

    tool = build_websearch(
        repository,
        WebSearchToolConfig.model_validate(
            {'kind': 'websearch', 'api_key': 'felo-key'},
        ),
    )

    assert tool.repository is repository
    assert tool.api_key.get_secret_value() == 'felo-key'
    assert tool.base_url == 'https://openapi.felo.ai/'
