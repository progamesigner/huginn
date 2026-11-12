import json
from base64 import b64encode

import httpx2
import pytest
from pydantic import SecretStr

from app.exts.transcription.clients import (
    ElevenLabsTranscriptionClient,
    GeminiTranscriptionClient,
    TranscriptionClient,
    filename,
)
from tests.helpers.http import MockHTTP

_URL = 'https://api.openai.com/v1/audio/transcriptions'
_VOICE = b'OggS not really an ogg'


def make_client(http: MockHTTP) -> TranscriptionClient:
    return TranscriptionClient(
        http=http.manager(),
        api_key=SecretStr('openai-key'),
        base_url='https://api.openai.com',
        model='gpt-4o-mini-transcribe',
    )


async def test_transcribe_uploads_the_bytes_and_returns_the_text(
    http: MockHTTP,
) -> None:
    route = http.post(_URL).mock(
        return_value=httpx2.Response(200, json={'text': 'buy milk on your way home'}),
    )

    text = await make_client(http).transcribe(_VOICE, 'audio/ogg')

    request = route.calls.last.request
    body = request.content.decode('utf-8', 'replace')
    assert request.headers['Authorization'] == 'Bearer openai-key'
    assert 'gpt-4o-mini-transcribe' in body
    assert 'filename="audio.ogg"' in body
    assert text == 'buy milk on your way home'


async def test_a_refused_upload_raises(http: MockHTTP) -> None:
    http.post(_URL).mock(return_value=httpx2.Response(400, json={'error': 'nope'}))

    with pytest.raises(httpx2.HTTPStatusError):
        await make_client(http).transcribe(_VOICE, 'audio/ogg')


@pytest.mark.parametrize(
    ('mime', 'expected'),
    [
        ('audio/ogg', 'audio.ogg'),
        ('audio/mpeg', 'audio.mp3'),
        ('audio/x-m4a', 'audio.m4a'),
        ('audio/flac', 'audio.flac'),
    ],
    ids=['ogg', 'mpeg', 'm4a', 'unmapped'],
)
def test_the_upload_is_named_for_the_format_openai_reads(
    mime: str,
    expected: str,
) -> None:
    # NOTE: telegram sends OGG/Opus and openai picks the decoder off the extension, so a
    # name of audio.mpeg (the subtype) would be refused where audio.mp3 is accepted
    assert filename(mime) == expected


async def test_gemini_sends_the_audio_inline_with_its_transcription_config(
    http: MockHTTP,
) -> None:
    route = http.post('https://gemini.example/v1beta/interactions').mock(
        return_value=httpx2.Response(
            200,
            json={
                'id': 'v1_x',
                'status': 'completed',
                'steps': [
                    {'type': 'user_input', 'content': [{'type': 'audio'}]},
                    {
                        'type': 'model_output',
                        'content': [
                            {'type': 'text', 'text': 'call me '},
                            {'type': 'text', 'text': 'back'},
                        ],
                    },
                ],
            },
        ),
    )
    client = GeminiTranscriptionClient(
        http=http.manager(),
        api_key=SecretStr('gemini-key'),
        base_url='https://gemini.example',
        model='gemini-3.5-transcribe',
        mode='verbatim',
        languages=('zh-TW',),
    )

    text = await client.transcribe(b'OggS voice', 'audio/ogg')

    assert text == 'call me back'
    request = route.calls.last.request
    assert request.headers['x-goog-api-key'] == 'gemini-key'
    assert json.loads(request.content) == {
        'model': 'gemini-3.5-transcribe',
        'input': [
            {
                'type': 'audio',
                'data': b64encode(b'OggS voice').decode(),
                'mime_type': 'audio/ogg',
            },
        ],
        'generation_config': {
            'transcription_config': {'mode': 'verbatim', 'language_codes': ['zh-TW']},
        },
    }


async def test_elevenlabs_uploads_the_audio_with_its_model_and_language(
    http: MockHTTP,
) -> None:
    route = http.post('https://eleven.example/v1/speech-to-text').mock(
        return_value=httpx2.Response(
            200,
            json={'language_code': 'zho', 'text': '回電給我', 'words': []},
        ),
    )
    client = ElevenLabsTranscriptionClient(
        http=http.manager(),
        api_key=SecretStr('eleven-key'),
        base_url='https://eleven.example',
        model='scribe_v2',
        language='zh',
    )

    text = await client.transcribe(b'OggS voice', 'audio/ogg')

    assert text == '回電給我'
    request = route.calls.last.request
    assert request.headers['xi-api-key'] == 'eleven-key'
    assert b'name="model_id"\r\n\r\nscribe_v2' in request.content
    assert b'name="language_code"\r\n\r\nzh' in request.content
    assert b'filename="audio.ogg"' in request.content


async def test_elevenlabs_leaves_the_language_to_detection_when_unset(
    http: MockHTTP,
) -> None:
    route = http.post('https://eleven.example/v1/speech-to-text').mock(
        return_value=httpx2.Response(200, json={'text': 'hi'}),
    )
    client = ElevenLabsTranscriptionClient(
        http=http.manager(),
        api_key=SecretStr('eleven-key'),
        base_url='https://eleven.example',
        model='scribe_v2',
    )

    await client.transcribe(b'OggS voice', 'audio/ogg')

    assert b'language_code' not in route.calls.last.request.content
