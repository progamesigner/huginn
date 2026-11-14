import json

import httpx2
import pytest
from pydantic import SecretStr

from app.exts.speech.clients import SpeechClient
from app.exts.speech.exceptions import SpeechSynthesisError
from tests.helpers.http import MockHTTP

_BASE_URL = 'https://speech.example'
_OGG = b'OggS opus voice'


def make_client(http: MockHTTP) -> SpeechClient:
    return SpeechClient(
        http=http.manager(),
        api_key=SecretStr('speech-key'),
        base_url=_BASE_URL,
        model='gpt-4o-mini-tts',
    )


async def test_synthesize_asks_for_the_format_and_returns_the_audio(
    http: MockHTTP,
) -> None:
    route = http.post(f'{_BASE_URL}/v1/audio/speech').mock(
        return_value=httpx2.Response(
            200,
            content=_OGG,
            headers={'content-type': 'audio/ogg'},
        ),
    )

    audio = await make_client(http).synthesize('hello there', 'alloy', 'opus')

    assert audio == _OGG
    request = route.calls.last.request
    assert request.headers['authorization'] == 'Bearer speech-key'
    assert json.loads(request.content) == {
        'model': 'gpt-4o-mini-tts',
        'input': 'hello there',
        'voice': 'alloy',
        'response_format': 'opus',
    }


async def test_synthesize_reports_a_refusal_with_the_vendors_reason(
    http: MockHTTP,
) -> None:
    http.post(f'{_BASE_URL}/v1/audio/speech').mock(
        return_value=httpx2.Response(
            400,
            json={'error': {'message': 'unknown voice'}},
        ),
    )

    with pytest.raises(SpeechSynthesisError, match=r'\(400\): unknown voice'):
        await make_client(http).synthesize('hello', 'nobody', 'opus')


async def test_gemini_speech_converts_raw_pcm_into_opus(http: MockHTTP) -> None:
    import shutil
    from base64 import b64encode

    from app.exts.speech.gemini import GeminiSpeechClient

    if shutil.which('ffmpeg') is None:
        pytest.skip('ffmpeg is not installed')
    route = http.post('https://speech.example/v1beta/models/test:generateContent').mock(
        return_value=httpx2.Response(
            200,
            json={
                'candidates': [
                    {
                        'finishReason': 'STOP',
                        'content': {
                            'parts': [
                                {
                                    'inlineData': {
                                        'mimeType': 'audio/L16;codec=pcm;rate=24000',
                                        'data': b64encode(b'\0\0' * 2400).decode(),
                                    },
                                },
                            ],
                        },
                    },
                ],
            },
        ),
    )
    client = GeminiSpeechClient(
        http.manager(),
        SecretStr('test-key'),
        'https://speech.example',
        'test',
    )
    encoded = await client.synthesize('Hello', 'Kore', 'opus')
    assert encoded.startswith(b'OggS')
    assert b'OpusHead' in encoded
    body = json.loads(route.calls.last.request.content)
    assert (
        body['generationConfig']['speechConfig']['voiceConfig']['prebuiltVoiceConfig'][
            'voiceName'
        ]
        == 'Kore'
    )


def test_gemini_speech_accepts_wav_and_rejects_invalid_pcm() -> None:
    import io
    import wave

    from app.exts.speech.gemini import _pcm

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b'\0\0' * 20)
    assert _pcm(buffer.getvalue(), 'audio/wav') == (b'\0\0' * 20, 24000)
    with pytest.raises(ValueError, match='invalid'):
        _pcm(b'x', 'audio/l16;rate=24000')
    with pytest.raises(ValueError, match='unsupported'):
        _pcm(b'xx', 'audio/mp3')
