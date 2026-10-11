import json
from base64 import b64encode
from collections.abc import AsyncIterator
from typing import override
from unittest.mock import Mock

import httpx
import pytest
import respx
from pydantic import SecretStr

from app.exts.images.clients import ImagesClient
from app.exts.images.exceptions import ImageGenerationError
from lib.http import HTTPClientManager

_BASE_URL = 'https://images.example'
_PNG = b'\x89PNG\r\n\x1a\n image'


def make_client(max_bytes: int = 20 * 1024 * 1024) -> ImagesClient:
    return ImagesClient(
        http=HTTPClientManager(),
        api_key=SecretStr('image-key'),
        base_url=_BASE_URL,
        model='gpt-image-1',
        max_bytes=max_bytes,
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


@pytest.mark.parametrize('size', [7, 8, 9, 10])
@respx.mock
async def test_inline_image_size_limit(size: int) -> None:
    data = b'x' * size
    respx.post(f'{_BASE_URL}/v1/images/generations').mock(
        return_value=httpx.Response(
            200,
            json={'data': [{'b64_json': b64encode(data).decode()}]},
        ),
    )

    if size <= 8:
        assert (await make_client(8).generate('a duck', '1024x1024')).data == data
    else:
        with pytest.raises(ImageGenerationError, match='size limit'):
            await make_client(8).generate('a duck', '1024x1024')


@respx.mock
async def test_oversized_base64_is_rejected_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    respx.post(f'{_BASE_URL}/v1/images/generations').mock(
        return_value=httpx.Response(
            200,
            json={'data': [{'b64_json': b64encode(b'x' * 10).decode()}]},
        ),
    )
    decode = Mock(side_effect=AssertionError('must not decode oversized input'))
    monkeypatch.setattr('app.exts.images.clients.b64decode', decode)

    with pytest.raises(ImageGenerationError, match='size limit'):
        await make_client(8).generate('a duck', '1024x1024')

    decode.assert_not_called()


class ImageStream(httpx.AsyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self.chunks = chunks
        self.read = 0
        self.closed = False

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.read += 1
            yield chunk

    @override
    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize('headers', [{}, {'content-length': '1'}])
@respx.mock
async def test_download_stops_and_closes_when_the_stream_exceeds_the_limit(
    headers: dict[str, str],
) -> None:
    stream = ImageStream((b'1234', b'56789', b'must not be read'))
    respx.get('https://storage.example/duck.png').mock(
        return_value=httpx.Response(200, headers=headers, stream=stream),
    )

    with pytest.raises(ImageGenerationError, match='size limit'):
        await make_client(8)._download('https://storage.example/duck.png')

    assert stream.read == 2
    assert stream.closed


@respx.mock
async def test_download_accepts_an_image_at_the_limit() -> None:
    stream = ImageStream((b'1234', b'5678'))
    respx.get('https://storage.example/duck.png').mock(
        return_value=httpx.Response(200, stream=stream),
    )

    assert (
        await make_client(8)._download('https://storage.example/duck.png')
        == b'12345678'
    )
    assert stream.closed
