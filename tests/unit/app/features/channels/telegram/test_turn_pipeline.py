import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import override
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from fastapi import FastAPI
from pydantic import SecretStr

from app.contracts.channels import ChannelReplyContract
from app.core.config.registry import HuginnConfig
from app.features.agents.config import AgentConfig, AgentConfigList
from app.features.agents.contracts import TurnHandlerContract
from app.features.agents.types import TurnRequest
from app.features.channels.config import ChannelConfigList, TranscriptionConfig
from app.features.channels.telegram.config import TelegramChannelConfig
from app.features.identities.contracts import IdentityResolverContract
from app.features.identities.types import AnonymousPrincipal, Principal
from bin.server import build
from lib.mcp import MCPClientContract
from lib.redis import RedisContract
from lib.router import RouterContract
from tests.helpers.logging import recording
from tests.helpers.redis import FakeClient

_BASE_URL = 'https://api.telegram.org/botbot-token'
_FILE_URL = 'https://api.telegram.org/file/botbot-token'
_CHAT_ID = 4242
_JPEG = b'\xff\xd8\xff\xe0 not really a jpeg'
_VOICE = b'OggS not really an ogg'
_HEADER = 'X-Telegram-Bot-Api-Secret-Token'
_PATH = '/webhooks/telegram'
_MILESTONES = (
    'Received telegram webhook',
    'Acknowledged',
    'Sent the first draft',
    'Finished the reply',
)


class StreamingTurns(TurnHandlerContract):
    done: asyncio.Event
    requests: list[TurnRequest]

    def __init__(self) -> None:
        self.done = asyncio.Event()
        self.requests = []

    @override
    async def handle(
        self,
        request: TurnRequest,
        reply: ChannelReplyContract,
    ) -> None:
        self.requests.append(request)
        try:
            await reply.on_text('the ')
            await reply.on_text('answer')
            await reply.finish()
        finally:
            self.done.set()


class FakeIdentities(IdentityResolverContract):
    @override
    async def resolve(
        self,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> Principal:
        return AnonymousPrincipal(provider, realm, external_user_id)


def update(update_id: int) -> dict[str, object]:
    return {
        'update_id': update_id,
        'message': {
            'message_id': update_id,
            'date': 1_721_600_000,
            'chat': {'id': _CHAT_ID, 'type': 'private'},
            'from': {'id': 77, 'is_bot': False, 'first_name': 'Han'},
            'text': 'hello',
        },
    }


def photo_update(update_id: int) -> dict[str, object]:
    return {
        'update_id': update_id,
        'message': {
            'message_id': update_id,
            'date': 1_721_600_000,
            'chat': {'id': _CHAT_ID, 'type': 'private'},
            'from': {'id': 77, 'is_bot': False, 'first_name': 'Han'},
            'photo': [
                {
                    'file_id': 'small',
                    'file_unique_id': 'u-small',
                    'width': 90,
                    'height': 60,
                    'file_size': 900,
                },
                {
                    'file_id': 'large',
                    'file_unique_id': 'u-large',
                    'width': 1280,
                    'height': 960,
                    'file_size': 2048,
                },
            ],
        },
    }


def voice_update(update_id: int) -> dict[str, object]:
    return {
        'update_id': update_id,
        'message': {
            'message_id': update_id,
            'date': 1_721_600_000,
            'chat': {'id': _CHAT_ID, 'type': 'private'},
            'from': {'id': 77, 'is_bot': False, 'first_name': 'Han'},
            'voice': {
                'file_id': 'note',
                'file_unique_id': 'u-note',
                'duration': 12,
                'mime_type': 'audio/ogg',
                'file_size': 1024,
            },
        },
    }


def make_config() -> HuginnConfig:
    channel = TelegramChannelConfig(
        kind='telegram',
        agent='scout',
        token=SecretStr('bot-token'),
        secret_token=SecretStr('sekret'),
    )
    return HuginnConfig(
        sections={
            'agents': AgentConfigList(
                {'scout': AgentConfig(description='scouts', instructions='be helpful')},
            ),
            'channels': ChannelConfigList({'support_bot': channel}),
            'transcription': TranscriptionConfig(),
        },
    )


def make_redis() -> MagicMock:
    # NOTE: the real actor runs here, so the double has to answer its lua as well as the
    # dedupe write: claim once, then drain nothing
    connection = MagicMock()
    connection.__aenter__ = AsyncMock(return_value=FakeClient(outcomes=[1, []]))
    connection.__aexit__ = AsyncMock(return_value=False)

    redis = MagicMock(spec=RedisContract)
    redis.connect.return_value = connection
    return redis


def message(message_id: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            'ok': True,
            'result': {
                'message_id': message_id,
                'date': 1_721_600_000,
                'chat': {'id': _CHAT_ID, 'type': 'private'},
            },
        },
    )


@asynccontextmanager
async def pipeline(turns: TurnHandlerContract) -> AsyncIterator[httpx.AsyncClient]:
    application = build()
    with (
        application.use(HuginnConfig, make_config()),
        application.use(MCPClientContract, MagicMock(spec=MCPClientContract)),
        application.use(RedisContract, make_redis()),
        application.use(TurnHandlerContract, turns),
        application.use(IdentityResolverContract, FakeIdentities()),
    ):
        app = application.resolve(RouterContract).app
        assert isinstance(app, FastAPI)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url='http://huginn',
            ) as client:
                yield client


@respx.mock
async def test_a_webhook_message_reaches_the_agent_turn_and_answers_in_chat() -> None:
    # NOTE: everything between the webhook and the vendor is real here -- dedupe, ack,
    # claim, the owner loop, the turn processor, routing policy and the streaming reply.
    # Only the model turn itself and the principal lookup are doubles.
    respx.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )
    send = respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    turns = StreamingTurns()

    async with pipeline(turns) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json=update(7),
        )
        assert response.status_code == 200
        async with asyncio.timeout(1):
            await turns.done.wait()

    assert json.loads(send.calls.last.request.content) == {
        'chat_id': str(_CHAT_ID),
        'text': 'the ',
    }
    assert json.loads(edit.calls.last.request.content)['text'] == 'the answer'

    request = turns.requests[0]
    assert [prompt.text for prompt in request.prompts] == ['hello']
    assert request.policy.agent == 'scout'
    assert request.prompts[0].actor.startswith('anonymous:')


@respx.mock
async def test_a_photo_with_no_caption_reaches_the_turn_with_its_bytes() -> None:
    respx.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )
    respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    file = respx.post(f'{_BASE_URL}/getFile').mock(
        return_value=httpx.Response(
            200,
            json={
                'ok': True,
                'result': {
                    'file_id': 'large',
                    'file_unique_id': 'u-large',
                    'file_size': 2048,
                    'file_path': 'photos/large.jpg',
                },
            },
        ),
    )
    download = respx.get(f'{_FILE_URL}/photos/large.jpg').mock(
        return_value=httpx.Response(200, content=_JPEG),
    )
    turns = StreamingTurns()

    async with pipeline(turns) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json=photo_update(8),
        )
        assert response.status_code == 200
        async with asyncio.timeout(1):
            await turns.done.wait()

    # NOTE: the largest size that fits the limit is the one worth downloading, and the
    # download is a second host away from the bot api
    assert json.loads(file.calls.last.request.content) == {'file_id': 'large'}
    assert download.called

    prompt = turns.requests[0].prompts[0]
    assert prompt.text == ''
    assert [(carried.kind, carried.data) for carried in prompt.attachments] == [
        ('image', _JPEG),
    ]
    assert prompt.attachments[0].hint == '[image: image/jpeg 1280x960]'


@respx.mock
async def test_a_voice_note_with_no_transcription_configured_is_answered_anyway() -> (
    None
):
    # NOTE: this config carries an empty transcription section, which is what a
    # deployment that never set one up has
    answered = asyncio.Event()

    def send(request: httpx.Request) -> httpx.Response:
        answered.set()
        return message(7)

    respx.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )
    sent = respx.post(f'{_BASE_URL}/sendMessage').mock(side_effect=send)
    respx.post(f'{_BASE_URL}/getFile').mock(
        return_value=httpx.Response(
            200,
            json={
                'ok': True,
                'result': {
                    'file_id': 'note',
                    'file_unique_id': 'u-note',
                    'file_size': 1024,
                    'file_path': 'voice/note.oga',
                },
            },
        ),
    )
    respx.get(f'{_FILE_URL}/voice/note.oga').mock(
        return_value=httpx.Response(200, content=_VOICE),
    )
    turns = StreamingTurns()

    async with pipeline(turns) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json=voice_update(9),
        )
        assert response.status_code == 200
        async with asyncio.timeout(1):
            await answered.wait()

    assert turns.requests == []
    assert json.loads(sent.calls.last.request.content) == {
        'chat_id': str(_CHAT_ID),
        'text': 'I could not read that voice message; please send it as text.',
    }


@respx.mock
async def test_one_round_logs_the_four_milestones_in_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    respx.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )
    respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    turns = StreamingTurns()

    with caplog.at_level('INFO'):
        async with pipeline(turns) as client:
            with recording(caplog):
                response = await client.post(
                    _PATH,
                    headers={_HEADER: 'sekret'},
                    json=update(7),
                )
                assert response.status_code == 200
                async with asyncio.timeout(1):
                    await turns.done.wait()

    # NOTE: these four are what the staging runbook measures a round by, so the order
    # they appear in is part of the contract, not an accident of where the calls sit
    assert [
        milestone
        for record in caplog.records
        for milestone in _MILESTONES
        if record.message.startswith(milestone)
    ] == list(_MILESTONES)
