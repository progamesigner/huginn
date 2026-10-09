import json
from base64 import b64encode

import httpx
import pytest
import respx
from pydantic import SecretStr

from app.exts.images.clients import ImagesClient
from app.exts.images.exceptions import ImageGenerationError
from lib.http import HTTPClientManager

_BASE_URL = 'https://images.example'
_PNG = b'\x89PNG\r\n\x1a\n image'


def make_client() -> ImagesClient:
    return ImagesClient(
        http=HTTPClientManager(),
        api_key=SecretStr('image-key'),
        base_url=_BASE_URL,
        model='gpt-image-1',
    )


@respx.mock
async def test_generate_decodes_the_inline_image() -> None:
    route = respx.post(f'{_BASE_URL}/v1/images/generations').mock(
        return_value=httpx.Response(
            200,
            json={'data': [{'b64_json': b64encode(_PNG).decode()}]},
        ),
    )

    generation = await make_client().generate('a duck', '1024x1024', 'high')

    assert generation.data == _PNG
    request = route.calls.last.request
    assert request.headers['authorization'] == 'Bearer image-key'
    assert json.loads(request.content) == {
        'model': 'gpt-image-1',
        'prompt': 'a duck',
        'size': '1024x1024',
        'n': 1,
        'quality': 'high',
    }


@respx.mock
async def test_generate_leaves_quality_to_the_vendor_when_unset() -> None:
    route = respx.post(f'{_BASE_URL}/v1/images/generations').mock(
        return_value=httpx.Response(
            200,
            json={'data': [{'b64_json': b64encode(_PNG).decode()}]},
        ),
    )

    await make_client().generate('a duck', '1024x1024')

    assert 'quality' not in json.loads(route.calls.last.request.content)


@respx.mock
async def test_generate_downloads_a_linked_image_without_the_api_key() -> None:
    respx.post(f'{_BASE_URL}/v1/images/generations').mock(
        return_value=httpx.Response(
            200,
            json={
                'data': [
                    {
                        'url': 'https://storage.example/duck.png',
                        'revised_prompt': 'a yellow duck',
                    },
                ],
            },
        ),
    )
    download = respx.get('https://storage.example/duck.png').mock(
        return_value=httpx.Response(200, content=_PNG),
    )

    generation = await make_client().generate('a duck', '1024x1024')

    assert generation.data == _PNG
    assert generation.revised_prompt == 'a yellow duck'
    assert 'authorization' not in download.calls.last.request.headers


@respx.mock
async def test_generate_refuses_a_link_over_plain_http() -> None:
    respx.post(f'{_BASE_URL}/v1/images/generations').mock(
        return_value=httpx.Response(
            200,
            json={'data': [{'url': 'http://storage.example/duck.png'}]},
        ),
    )

    with pytest.raises(ImageGenerationError, match='plain http'):
        await make_client().generate('a duck', '1024x1024')


@respx.mock
async def test_generate_reports_a_refusal_with_the_vendors_reason() -> None:
    respx.post(f'{_BASE_URL}/v1/images/generations').mock(
        return_value=httpx.Response(
            400,
            json={'error': {'message': 'rejected by the safety system'}},
        ),
    )

    with pytest.raises(
        ImageGenerationError,
        match=r'refused the request \(400\): rejected by the safety system',
    ):
        await make_client().generate('a duck', '1024x1024')


@respx.mock
async def test_generate_lets_a_server_error_surface() -> None:
    respx.post(f'{_BASE_URL}/v1/images/generations').mock(
        return_value=httpx.Response(503),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await make_client().generate('a duck', '1024x1024')
