import json

import httpx
import pytest
import respx
from pydantic import SecretStr

from app.exts.speech.clients import SpeechClient
from app.exts.speech.exceptions import SpeechSynthesisError
from lib.http import HTTPClientManager

_BASE_URL = 'https://speech.example'
_OGG = b'OggS opus voice'


def make_client() -> SpeechClient:
    return SpeechClient(
        http=HTTPClientManager(),
        api_key=SecretStr('speech-key'),
        base_url=_BASE_URL,
        model='gpt-4o-mini-tts',
    )


@respx.mock
async def test_synthesize_asks_for_the_format_and_returns_the_audio() -> None:
    route = respx.post(f'{_BASE_URL}/v1/audio/speech').mock(
        return_value=httpx.Response(
            200,
            content=_OGG,
            headers={'content-type': 'audio/ogg'},
        ),
    )

    audio = await make_client().synthesize('hello there', 'alloy', 'opus')

    assert audio == _OGG
    request = route.calls.last.request
    assert request.headers['authorization'] == 'Bearer speech-key'
    assert json.loads(request.content) == {
        'model': 'gpt-4o-mini-tts',
        'input': 'hello there',
        'voice': 'alloy',
        'response_format': 'opus',
    }


@respx.mock
async def test_synthesize_reports_a_refusal_with_the_vendors_reason() -> None:
    respx.post(f'{_BASE_URL}/v1/audio/speech').mock(
        return_value=httpx.Response(
            400,
            json={'error': {'message': 'unknown voice'}},
        ),
    )

    with pytest.raises(SpeechSynthesisError, match=r'\(400\): unknown voice'):
        await make_client().synthesize('hello', 'nobody', 'opus')
