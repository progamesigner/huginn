import httpx
import pytest
import respx
from pydantic import HttpUrl, SecretStr

from app.features.channels.config import TranscriptionConfig
from app.features.channels.exceptions import TranscriptionUnavailableError
from app.features.channels.repositories import TranscriptionRepository
from lib.http import HTTPClientManager

_VOICE = b'OggS not really an ogg'


def make_repository(config: TranscriptionConfig) -> TranscriptionRepository:
    return TranscriptionRepository(HTTPClientManager(), config)


def configured() -> TranscriptionConfig:
    return TranscriptionConfig(
        kind='openai',
        model='gpt-4o-mini-transcribe',
        api_key=SecretStr('openai-key'),
        base_url=HttpUrl('https://transcribe.example'),
    )


@respx.mock
async def test_transcribe_answers_with_what_the_vendor_heard() -> None:
    route = respx.post('https://transcribe.example/v1/audio/transcriptions').mock(
        return_value=httpx.Response(200, json={'text': 'call me back'}),
    )

    text = await make_repository(configured()).transcribe('audio/ogg', _VOICE)

    assert route.called
    assert text == 'call me back'


async def test_an_unconfigured_section_reports_itself_as_unavailable() -> None:
    # NOTE: this is the difference the round has to act on -- nobody configured
    # transcription, so no retry will ever make this voice message readable
    with pytest.raises(TranscriptionUnavailableError):
        await make_repository(TranscriptionConfig()).transcribe('audio/ogg', _VOICE)
