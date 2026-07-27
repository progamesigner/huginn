import json

import httpx
import pytest
import respx
from pydantic import SecretStr

from app.exts.felo.clients import FeloClient, SearchResource
from lib.http import HTTPClientManager

_PAYLOAD = {
    'status': 'ok',
    'data': {
        'answer': 'Taipei is sunny today.',
        'query_analysis': {'queries': ['taipei weather today']},
        'resources': [
            {
                'link': 'https://weather.example/taipei',
                'title': 'Taipei Weather',
                'snippet': 'Sunny, 31°C.',
            },
        ],
    },
}


def make_client() -> FeloClient:
    return FeloClient(
        http=HTTPClientManager(),
        api_key=SecretStr('felo-key'),
        base_url='https://openapi.felo.ai',
    )


@respx.mock
async def test_search_parses_answer_and_resources() -> None:
    route = respx.post('https://openapi.felo.ai/v2/chat').mock(
        return_value=httpx.Response(200, json=_PAYLOAD),
    )

    result = await make_client().search('taipei weather')

    request = route.calls.last.request
    assert request.headers['Authorization'] == 'Bearer felo-key'
    assert json.loads(request.content) == {'query': 'taipei weather'}
    assert result.answer == 'Taipei is sunny today.'
    assert result.resources == (
        SearchResource(
            link='https://weather.example/taipei',
            title='Taipei Weather',
            snippet='Sunny, 31°C.',
        ),
    )


@respx.mock
async def test_search_ignores_unknown_vendor_fields() -> None:
    payload = {
        'status': 'ok',
        'id': 'task-1',
        'message_id': 'message-1',
        'data': {
            'answer': 'An answer.',
            'confidence': 0.9,
            'resources': [
                {
                    'link': 'https://example.com',
                    'title': 'Example',
                    'score': 42,
                },
            ],
        },
    }
    respx.post('https://openapi.felo.ai/v2/chat').mock(
        return_value=httpx.Response(200, json=payload),
    )

    result = await make_client().search('anything')

    assert result.answer == 'An answer.'
    assert result.resources[0].snippet == ''


@respx.mock
async def test_search_raises_on_http_error() -> None:
    respx.post('https://openapi.felo.ai/v2/chat').mock(
        return_value=httpx.Response(500),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await make_client().search('anything')


@respx.mock
async def test_search_raises_on_timeout() -> None:
    respx.post('https://openapi.felo.ai/v2/chat').mock(
        side_effect=httpx.ReadTimeout('timed out'),
    )

    with pytest.raises(httpx.TimeoutException):
        await make_client().search('anything')
