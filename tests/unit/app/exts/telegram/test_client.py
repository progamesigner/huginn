import json

import httpx
import pytest
import respx
from pydantic import SecretStr

from app.exts.telegram.client import TelegramClient
from lib.http import HTTPClientManager

_BASE_URL = 'https://api.telegram.org/bottelegram-token'
_MESSAGE = {
    'message_id': 7,
    'date': 1_721_600_000,
    'chat': {'id': 123, 'type': 'private'},
    'text': 'hello',
}


def make_client() -> TelegramClient:
    return TelegramClient(
        http=HTTPClientManager(),
        bot_token=SecretStr('telegram-token'),
    )


@respx.mock
async def test_send_message() -> None:
    route = respx.post(f'{_BASE_URL}/sendMessage').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': _MESSAGE}),
    )

    message = await make_client().send_message(123, 'hello')

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': 123,
        'text': 'hello',
    }
    assert message.message_id == 7


@respx.mock
async def test_edit_message_text() -> None:
    route = respx.post(f'{_BASE_URL}/editMessageText').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': _MESSAGE}),
    )

    message = await make_client().edit_message_text(123, 7, 'updated')

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': 123,
        'message_id': 7,
        'text': 'updated',
    }
    assert message.message_id == 7


@respx.mock
async def test_set_message_reaction() -> None:
    route = respx.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )

    result = await make_client().set_message_reaction(123, 7, '👀')

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': 123,
        'message_id': 7,
        'reaction': [{'type': 'emoji', 'emoji': '👀'}],
    }
    assert result is True


@respx.mock
async def test_get_file() -> None:
    route = respx.post(f'{_BASE_URL}/getFile').mock(
        return_value=httpx.Response(
            200,
            json={
                'ok': True,
                'result': {
                    'file_id': 'voice-file',
                    'file_unique_id': 'voice-unique',
                    'file_size': 512,
                    'file_path': 'voice/file.oga',
                },
            },
        ),
    )

    file = await make_client().get_file('voice-file')

    assert json.loads(route.calls.last.request.content) == {'file_id': 'voice-file'}
    assert file.file_path == 'voice/file.oga'


@respx.mock
async def test_retries_rate_limit_after_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr('app.exts.telegram.client.asyncio.sleep', record_sleep)
    route = respx.post(f'{_BASE_URL}/sendMessage').mock(
        side_effect=[
            httpx.Response(
                429,
                json={
                    'ok': False,
                    'error_code': 429,
                    'parameters': {'retry_after': 3},
                },
            ),
            httpx.Response(200, json={'ok': True, 'result': _MESSAGE}),
        ],
    )

    await make_client().send_message(123, 'hello')

    assert route.call_count == 2
    assert sleeps == [3.0]


@respx.mock
async def test_retries_server_errors_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr('app.exts.telegram.client.asyncio.sleep', no_sleep)
    route = respx.post(f'{_BASE_URL}/sendMessage').mock(
        return_value=httpx.Response(503),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await make_client().send_message(123, 'hello')

    assert route.call_count == 3
