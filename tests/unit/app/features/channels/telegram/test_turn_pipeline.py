import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import override
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from app.contracts.channels import ChannelReplyContract
from app.core.config.registry import HuginnConfig
from app.features.agents.config import AgentConfig, AgentConfigList
from app.features.agents.contracts import TurnServiceContract
from app.features.agents.entities import TurnRequest
from app.features.channels.config import ChannelConfigList, TranscriptionConfig
from app.features.channels.telegram.config import TelegramChannelConfig
from app.features.identities.contracts import IdentityResolutionServiceContract
from app.features.identities.entities import AnonymousPrincipal, Principal
from bin.server import build
from lib.http import HTTPClientContract
from lib.mcp import MCPClientContract
from lib.redis import RedisContract
from lib.router import RouterContract
from tests.helpers.http import MockHTTP
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


class StreamingTurns(TurnServiceContract):
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
            # NOTE: the real handler reports the routed agent before it calls the model,
            # and the reply hangs its typing status off that
            await reply.on_routed('scout')
            # NOTE: a real model call waits on the network, which is what lets the typing
            # task have its turn; this fake answers from memory, so it yields once by hand
            await asyncio.sleep(0)
            await reply.on_text('the ')
            await reply.on_text('answer')
            await reply.finish()
        finally:
            self.done.set()


class FakeIdentities(IdentityResolutionServiceContract):
    @override
    async def resolve(
        self,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> Principal:
        return AnonymousPrincipal(
            provider=provider,
            realm=realm,
            external_user_id=external_user_id,
        )


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


def message(message_id: int) -> httpx2.Response:
    return httpx2.Response(
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
async def pipeline(
    http: MockHTTP,
    turns: TurnServiceContract,
) -> AsyncIterator[httpx2.AsyncClient]:
    for method in ('setMyCommands', 'deleteMyCommands'):
        http.post(f'{_BASE_URL}/{method}').mock(
            return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
        )
    application = build()
    with (
        application.use(HuginnConfig, make_config()),
        application.use(HTTPClientContract, http.manager()),
        application.use(MCPClientContract, MagicMock(spec=MCPClientContract)),
        application.use(RedisContract, make_redis()),
        application.use(TurnServiceContract, turns),
        application.use(IdentityResolutionServiceContract, FakeIdentities()),
    ):
        app = application.resolve(RouterContract).app
        assert isinstance(app, FastAPI)
        async with app.router.lifespan_context(app):
            transport = httpx2.ASGITransport(app=app)
            async with httpx2.AsyncClient(
                transport=transport,
                base_url='http://huginn',
            ) as client:
                yield client


async def test_a_webhook_message_reaches_the_agent_turn_and_answers_in_chat(
    http: MockHTTP,
) -> None:
    # NOTE: everything between the webhook and the vendor is real here -- dedupe, ack,
    # claim, the owner loop, the turn processor, routing policy and the streaming reply.
    # Only the model turn itself and the principal lookup are doubles.
    http.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )
    draft = http.post(f'{_BASE_URL}/sendMessageDraft').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    action = http.post(f'{_BASE_URL}/sendChatAction').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )
    turns = StreamingTurns()

    async with pipeline(http, turns) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json=update(7),
        )
        assert response.status_code == 200
        async with asyncio.timeout(1):
            await turns.done.wait()

    # NOTE: the update comes from a private chat, so the reply streams through draft
    # previews and lands as a single message instead of one that is edited in place
    assert all(
        json.loads(call.request.content)['chat_id'] == str(_CHAT_ID)
        for call in draft.calls
    )
    assert json.loads(send.calls.last.request.content) == {
        'chat_id': str(_CHAT_ID),
        'text': 'the answer',
    }
    assert send.call_count == 1
    assert not edit.called
    assert json.loads(action.calls.last.request.content) == {
        'chat_id': str(_CHAT_ID),
        'action': 'typing',
    }

    request = turns.requests[0]
    assert [prompt.text for prompt in request.prompts] == ['hello']
    assert request.policy.agent == 'scout'
    assert request.prompts[0].actor.startswith('anonymous:')


async def test_a_photo_with_no_caption_reaches_the_turn_with_its_bytes(
    http: MockHTTP,
) -> None:
    http.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )
    http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    http.post(f'{_BASE_URL}/sendChatAction').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )
    file = http.post(f'{_BASE_URL}/getFile').mock(
        return_value=httpx2.Response(
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
    download = http.get(f'{_FILE_URL}/photos/large.jpg').mock(
        return_value=httpx2.Response(200, content=_JPEG),
    )
    turns = StreamingTurns()

    async with pipeline(http, turns) as client:
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


async def test_a_voice_note_with_no_transcription_configured_is_answered_anyway(
    http: MockHTTP,
) -> None:
    # NOTE: this config carries an empty transcription section, which is what a
    # deployment that never set one up has
    answered = asyncio.Event()

    def send(request: httpx2.Request) -> httpx2.Response:
        answered.set()
        return message(7)

    http.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )
    sent = http.post(f'{_BASE_URL}/sendMessage').mock(side_effect=send)
    http.post(f'{_BASE_URL}/getFile').mock(
        return_value=httpx2.Response(
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
    http.get(f'{_FILE_URL}/voice/note.oga').mock(
        return_value=httpx2.Response(200, content=_VOICE),
    )
    turns = StreamingTurns()

    async with pipeline(http, turns) as client:
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


async def test_one_round_logs_the_four_milestones_in_order(
    http: MockHTTP,
    caplog: pytest.LogCaptureFixture,
) -> None:
    http.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )
    http.post(f'{_BASE_URL}/sendMessageDraft').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )
    http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    http.post(f'{_BASE_URL}/sendChatAction').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )
    turns = StreamingTurns()

    with caplog.at_level('INFO'):
        async with pipeline(http, turns) as client:
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
