from datetime import timedelta

import httpx
import pytest
import respx
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
from lib.http import HTTPClientManager
from lib.llm import ToolCallSegment
from tests.helpers.blobs import MemoryBlobs
from tests.helpers.clock import FakeClock
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


def make_tool(blobs: MemoryBlobs, **overrides: object) -> SpeechTool:
    return build_speech(
        SpeechRepository(HTTPClientManager()),
        ArtifactWriter(blobs, _TTL),
        config(**overrides),
    )


def vendor() -> respx.Route:
    return respx.post(f'{_VENDOR}/v1/audio/speech').mock(
        return_value=httpx.Response(200, content=_OGG),
    )


@respx.mock
async def test_spoken_text_becomes_an_ogg_voice_note() -> None:
    route = vendor()
    blobs = MemoryBlobs()

    result = await make_tool(blobs).run(SpeechParams(text='hello there'))

    assert result.text == 'Spoke 11 characters as a voice message.'
    (artifact,) = result.artifacts
    assert (artifact.kind, artifact.mime, artifact.name) == (
        'audio',
        'audio/ogg',
        'speech.ogg',
    )
    assert artifact.data == _OGG
    assert b'"voice":"alloy"' in route.calls.last.request.content.replace(b' ', b'')


@respx.mock
async def test_the_model_may_pick_another_voice() -> None:
    route = vendor()

    await make_tool(MemoryBlobs()).run(SpeechParams(text='hi', voice='nova'))

    assert b'"voice":"nova"' in route.calls.last.request.content.replace(b' ', b'')


@respx.mock
async def test_an_mp3_voice_is_labelled_as_mp3() -> None:
    vendor()

    result = await make_tool(MemoryBlobs(), format='mp3').run(SpeechParams(text='hi'))

    assert (result.artifacts[0].mime, result.artifacts[0].name) == (
        'audio/mpeg',
        'speech.mp3',
    )


@respx.mock
async def test_text_over_the_limit_is_refused_without_calling_the_vendor() -> None:
    route = vendor()
    tools = ToolSet(
        {'narrator': make_tool(MemoryBlobs(), max_chars=5)},
        logger=make_logger(),
    )

    result, artifacts = await tools.execute(
        ToolCallSegment(id='c', name='narrator', arguments='{"text": "too long"}'),
    )

    assert result.error is True
    assert 'at most 5' in result.content
    assert artifacts == ()
    assert not route.called


def test_build_speech_refuses_another_tools_config() -> None:
    other = WebSearchToolConfig.model_validate({'kind': 'websearch', 'api_key': 'k'})

    with pytest.raises(ToolError, match='SpeechToolConfig'):
        build_speech(
            SpeechRepository(HTTPClientManager()),
            ArtifactWriter(MemoryBlobs(), _TTL),
            other,
        )


@respx.mock
async def test_a_voice_note_reaches_telegram_as_a_voice_bubble() -> None:
    vendor()
    voice = respx.post(f'{_TELEGRAM}/sendVoice').mock(
        return_value=httpx.Response(
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
    audio = respx.post(f'{_TELEGRAM}/sendAudio')
    result = await make_tool(MemoryBlobs()).run(SpeechParams(text='hello'))
    reply = TelegramReply(
        TelegramClient(http=HTTPClientManager(), bot_token=SecretStr('telegram-token')),
        5,
        FakeClock(),
        logger=make_logger(),
    )

    await reply.on_artifact(result.artifacts[0])

    assert voice.called
    assert not audio.called
