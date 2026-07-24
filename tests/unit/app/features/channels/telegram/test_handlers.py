import json
import time

import httpx
import respx
from pydantic import SecretStr

from app.exts.telegram.types import Update
from app.features.channels.telegram.config import TelegramChannelConfig
from app.features.channels.telegram.handlers import ReactionUpdateHandler
from lib.http import HTTPClientManager

_BASE_URL = 'https://api.telegram.org/botbot-token'
_MESSAGE = {
    'message_id': 42,
    'date': 1_721_600_000,
    'chat': {'id': 123, 'type': 'private'},
    'text': 'hello',
}


def make_handler() -> ReactionUpdateHandler:
    return ReactionUpdateHandler(HTTPClientManager())


def make_channel() -> TelegramChannelConfig:
    return TelegramChannelConfig(
        kind='telegram',
        token=SecretStr('bot-token'),
        secret_token=SecretStr('sekret'),
    )


@respx.mock
async def test_reacts_to_an_incoming_message() -> None:
    route = respx.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )

    update = Update.model_validate({'update_id': 7, 'message': _MESSAGE})
    await make_handler().handle('support_bot', make_channel(), update)

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': 123,
        'message_id': 42,
        'reaction': [{'type': 'emoji', 'emoji': '👀'}],
    }


@respx.mock
async def test_ignores_updates_without_a_message() -> None:
    route = respx.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )

    update = Update.model_validate({'update_id': 7})
    await make_handler().handle('support_bot', make_channel(), update)

    assert not route.called


@respx.mock
async def test_acks_within_the_latency_budget() -> None:
    fired: list[float] = []

    def _record(request: httpx.Request) -> httpx.Response:
        fired.append(time.perf_counter())
        return httpx.Response(200, json={'ok': True, 'result': True})

    respx.post(f'{_BASE_URL}/setMessageReaction').mock(side_effect=_record)

    update = Update.model_validate({'update_id': 7, 'message': _MESSAGE})
    start = time.perf_counter()
    await make_handler().handle('support_bot', make_channel(), update)

    assert fired
    assert fired[0] - start < 0.3
