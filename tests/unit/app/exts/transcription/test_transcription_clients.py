import httpx
import pytest
import respx
from pydantic import SecretStr

from app.exts.transcription.clients import TranscriptionClient, filename
from lib.http import HTTPClientManager

_URL = 'https://api.openai.com/v1/audio/transcriptions'
_VOICE = b'OggS not really an ogg'


def make_client() -> TranscriptionClient:
    return TranscriptionClient(
        http=HTTPClientManager(),
        api_key=SecretStr('openai-key'),
        base_url='https://api.openai.com',
        model='gpt-4o-mini-transcribe',
    )


@respx.mock
async def test_transcribe_uploads_the_bytes_and_returns_the_text() -> None:
    route = respx.post(_URL).mock(
        return_value=httpx.Response(200, json={'text': 'buy milk on your way home'}),
    )

    text = await make_client().transcribe(_VOICE, 'audio/ogg')

    request = route.calls.last.request
    body = request.content.decode('utf-8', 'replace')
    assert request.headers['Authorization'] == 'Bearer openai-key'
    assert 'gpt-4o-mini-transcribe' in body
    assert 'filename="audio.ogg"' in body
    assert text == 'buy milk on your way home'


@respx.mock
async def test_a_refused_upload_raises() -> None:
    respx.post(_URL).mock(return_value=httpx.Response(400, json={'error': 'nope'}))

    with pytest.raises(httpx.HTTPStatusError):
        await make_client().transcribe(_VOICE, 'audio/ogg')


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
