import json

import httpx2
import pytest
from pydantic import SecretStr

from app.exts.felo.clients import FeloClient, SearchResource
from tests.helpers.http import MockHTTP

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


def make_client(http: MockHTTP) -> FeloClient:
    return FeloClient(
        http=http.manager(),
        api_key=SecretStr('felo-key'),
        base_url='https://openapi.felo.ai',
        timeout=60.0,
    )


async def test_search_parses_answer_and_resources(http: MockHTTP) -> None:
    route = http.post('https://openapi.felo.ai/v2/chat').mock(
        return_value=httpx2.Response(200, json=_PAYLOAD),
    )

    result = await make_client(http).search('taipei weather')

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


async def test_search_ignores_unknown_vendor_fields(http: MockHTTP) -> None:
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
    http.post('https://openapi.felo.ai/v2/chat').mock(
        return_value=httpx2.Response(200, json=payload),
    )

    result = await make_client(http).search('anything')

    assert result.answer == 'An answer.'
    assert result.resources[0].snippet == ''


async def test_search_raises_on_http_error(http: MockHTTP) -> None:
    http.post('https://openapi.felo.ai/v2/chat').mock(
        return_value=httpx2.Response(500),
    )

    with pytest.raises(httpx2.HTTPStatusError):
        await make_client(http).search('anything')


async def test_search_raises_on_timeout(http: MockHTTP) -> None:
    http.post('https://openapi.felo.ai/v2/chat').mock(
        error=httpx2.ReadTimeout('timed out'),
    )

    with pytest.raises(httpx2.TimeoutException):
        await make_client(http).search('anything')


async def test_search_waits_for_the_configured_timeout(http: MockHTTP) -> None:
    route = http.post('https://openapi.felo.ai/v2/chat').mock(
        return_value=httpx2.Response(200, json=_PAYLOAD),
    )
    client = FeloClient(
        http=http.manager(timeout=30.0),
        api_key=SecretStr('felo-key'),
        base_url='https://openapi.felo.ai',
        timeout=75.0,
    )

    await client.search('anything')

    assert route.calls.last.request.extensions['timeout'] == {
        'connect': 75.0,
        'read': 75.0,
        'write': 75.0,
        'pool': 75.0,
    }
