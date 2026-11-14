from datetime import timedelta

import httpx2
import pytest
from pydantic import SecretStr

from app.exts.telegram.clients import TelegramClient
from app.exts.telegram.gateways import TelegramReply
from app.features.agents.tooling import ToolSet
from app.features.tools.artifacts import ArtifactWriter
from app.features.tools.exceptions import ToolError
from app.features.tools.speech.config import SpeechToolConfig
from app.features.tools.speech.repositories import SpeechRepository
from app.features.tools.speech.tool import SpeechParams, SpeechTool, build_speech
from app.features.tools.websearch.config import WebSearchToolConfig
from lib.llm import ToolCallSegment
from tests.helpers.blobs import MemoryBlobs
from tests.helpers.clock import FakeClock
from tests.helpers.http import MockHTTP, Route
from tests.helpers.logging import make_logger

_VENDOR = 'https://speech.example'
_TELEGRAM = 'https://api.telegram.org/bottelegram-token'
_OGG = b'OggS opus voice'
_TTL = timedelta(days=30)


def config(**overrides: object) -> SpeechToolConfig:
    return SpeechToolConfig.model_validate(
        {
            'kind': 'speech',
            'model': 'gpt-4o-mini-tts',
            'voice': 'alloy',
            'api_key': 'speech-key',
            'base_url': _VENDOR,
            **overrides,
        },
    )


def make_tool(http: MockHTTP, blobs: MemoryBlobs, **overrides: object) -> SpeechTool:
    return build_speech(
        SpeechRepository(http.manager()),
        ArtifactWriter(blobs, _TTL),
        config(**overrides),
    )


def vendor(http: MockHTTP) -> Route:
    return http.post(f'{_VENDOR}/v1/audio/speech').mock(
        return_value=httpx2.Response(200, content=_OGG),
    )


async def test_spoken_text_becomes_an_ogg_voice_note(http: MockHTTP) -> None:
    route = vendor(http)
    blobs = MemoryBlobs()

    result = await make_tool(http, blobs).run(SpeechParams(text='hello there'))

    assert result.text == 'Spoke 11 characters as a voice message.'
    (artifact,) = result.artifacts
    assert (artifact.kind, artifact.mime, artifact.name) == (
        'audio',
        'audio/ogg',
        'speech.ogg',
    )
    assert artifact.data == _OGG
    assert b'"voice":"alloy"' in route.calls.last.request.content.replace(b' ', b'')


async def test_the_model_may_pick_another_voice(http: MockHTTP) -> None:
    route = vendor(http)

    await make_tool(http, MemoryBlobs()).run(SpeechParams(text='hi', voice='nova'))

    assert b'"voice":"nova"' in route.calls.last.request.content.replace(b' ', b'')


async def test_an_mp3_voice_is_labelled_as_mp3(http: MockHTTP) -> None:
    vendor(http)

    result = await make_tool(http, MemoryBlobs(), format='mp3').run(
        SpeechParams(text='hi'),
    )

    assert (result.artifacts[0].mime, result.artifacts[0].name) == (
        'audio/mpeg',
        'speech.mp3',
    )


async def test_text_over_the_limit_is_refused_without_calling_the_vendor(
    http: MockHTTP,
) -> None:
    route = vendor(http)
    tools = ToolSet(
        {'narrator': make_tool(http, MemoryBlobs(), max_chars=5)},
        logger=make_logger(),
    )

    result, artifacts = await tools.execute(
        ToolCallSegment(id='c', name='narrator', arguments='{"text": "too long"}'),
    )

    assert result.error is True
    assert 'at most 5' in result.content
    assert artifacts == ()
    assert not route.called


def test_build_speech_refuses_another_tools_config(http: MockHTTP) -> None:
    other = WebSearchToolConfig.model_validate({'kind': 'websearch', 'api_key': 'k'})

    with pytest.raises(ToolError, match='SpeechToolConfig'):
        build_speech(
            SpeechRepository(http.manager()),
            ArtifactWriter(MemoryBlobs(), _TTL),
            other,
        )


async def test_a_voice_note_reaches_telegram_as_a_voice_bubble(http: MockHTTP) -> None:
    vendor(http)
    voice = http.post(f'{_TELEGRAM}/sendVoice').mock(
        return_value=httpx2.Response(
            200,
            json={
                'ok': True,
                'result': {
                    'message_id': 2,
                    'date': 1_721_600_000,
                    'chat': {'id': 5, 'type': 'private'},
                },
            },
        ),
    )
    audio = http.post(f'{_TELEGRAM}/sendAudio')
    result = await make_tool(http, MemoryBlobs()).run(SpeechParams(text='hello'))
    reply = TelegramReply(
        TelegramClient(http=http.manager(), bot_token=SecretStr('telegram-token')),
        5,
        FakeClock(),
        logger=make_logger(),
    )

    await reply.on_artifact(result.artifacts[0])

    assert voice.called
    assert not audio.called


def test_gemini_speech_uses_google_defaults_and_requires_opus() -> None:
    import pytest
    from pydantic import ValidationError

    from app.features.tools.speech.config import SpeechToolConfig

    config = SpeechToolConfig(
        kind='speech',
        vendor='gemini',
        model='test',
        voice='Kore',
        api_key=SecretStr('test'),
    )
    assert str(config.base_url) == 'https://generativelanguage.googleapis.com/'
    with pytest.raises(ValidationError, match='opus'):
        SpeechToolConfig(
            kind='speech',
            vendor='gemini',
            model='test',
            voice='Kore',
            api_key=SecretStr('test'),
            format='wav',
        )
