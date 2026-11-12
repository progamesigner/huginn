import httpx2
import pytest
from pydantic import HttpUrl, SecretStr

from app.features.channels.config import TranscriptionConfig, TranscriptionKind
from app.features.channels.exceptions import TranscriptionUnavailableError
from app.features.channels.repositories import TranscriptionRepository
from tests.helpers.http import MockHTTP

_VOICE = b'OggS not really an ogg'


def make_repository(
    http: MockHTTP,
    config: TranscriptionConfig,
) -> TranscriptionRepository:
    return TranscriptionRepository(http.manager(), config)


def configured() -> TranscriptionConfig:
    return TranscriptionConfig(
        kind='openai',
        model='gpt-4o-mini-transcribe',
        api_key=SecretStr('openai-key'),
        base_url=HttpUrl('https://transcribe.example'),
    )


async def test_transcribe_answers_with_what_the_vendor_heard(http: MockHTTP) -> None:
    route = http.post('https://transcribe.example/v1/audio/transcriptions').mock(
        return_value=httpx2.Response(200, json={'text': 'call me back'}),
    )

    text = await make_repository(http, configured()).transcribe('audio/ogg', _VOICE)

    assert route.called
    assert text == 'call me back'


async def test_an_unconfigured_section_reports_itself_as_unavailable(
    http: MockHTTP,
) -> None:
    # NOTE: this is the difference the round has to act on -- nobody configured
    # transcription, so no retry will ever make this voice message readable
    with pytest.raises(TranscriptionUnavailableError):
        await make_repository(http, TranscriptionConfig()).transcribe(
            'audio/ogg',
            _VOICE,
        )


@pytest.mark.parametrize(
    ('kind', 'url', 'answer'),
    [
        (
            'openai',
            'https://api.openai.com/v1/audio/transcriptions',
            {'text': 'heard'},
        ),
        (
            'gemini',
            'https://generativelanguage.googleapis.com/v1beta/interactions',
            {
                'steps': [
                    {
                        'type': 'model_output',
                        'content': [{'type': 'text', 'text': 'heard'}],
                    },
                ],
            },
        ),
        (
            'elevenlabs',
            'https://api.elevenlabs.io/v1/speech-to-text',
            {'text': 'heard'},
        ),
    ],
)
async def test_each_kind_reaches_its_own_vendor_by_default(
    http: MockHTTP,
    kind: TranscriptionKind,
    url: str,
    answer: dict[str, object],
) -> None:
    route = http.post(url).mock(return_value=httpx2.Response(200, json=answer))
    config = TranscriptionConfig(kind=kind, model='m', api_key=SecretStr('k'))

    assert (
        await make_repository(http, config).transcribe('audio/ogg', _VOICE) == 'heard'
    )
    assert route.called


def test_a_config_names_every_supported_kind_and_its_options() -> None:
    config = TranscriptionConfig.model_validate_json(
        '{"kind": "gemini", "model": "gemini-3.5-transcribe", "api_key": "k",'
        ' "mode": "verbatim", "language_codes": ["zh-TW", "en-US"]}',
    )

    assert (config.kind, config.mode, config.language_codes) == (
        'gemini',
        'verbatim',
        ('zh-TW', 'en-US'),
    )
    with pytest.raises(ValueError, match='kind'):
        TranscriptionConfig.model_validate_json('{"kind": "aws"}')
