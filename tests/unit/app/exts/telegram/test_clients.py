import json

import httpx
import pytest
import respx
from pydantic import SecretStr

from app.exts.telegram.clients import TelegramClient
from app.exts.telegram.exceptions import MediaTooLargeError, MediaUnavailableError
from lib.http import HTTPClientManager

_BASE_URL = 'https://api.telegram.org/bottelegram-token'
_FILE_URL = 'https://api.telegram.org/file/bottelegram-token'
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


def file_response(
    *,
    file_size: int | None = 512,
    file_path: str | None = 'photos/file.jpg',
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            'ok': True,
            'result': {
                'file_id': 'photo-file',
                'file_unique_id': 'photo-unique',
                'file_size': file_size,
                'file_path': file_path,
            },
        },
    )


@respx.mock
async def test_download_file_reads_raw_bytes_from_the_file_host() -> None:
    route = respx.get(f'{_FILE_URL}/photos/file.jpg').mock(
        return_value=httpx.Response(200, content=b'\xff\xd8jpeg'),
    )

    content = await make_client().download_file('photos/file.jpg')

    assert route.called
    assert content == b'\xff\xd8jpeg'


@respx.mock
async def test_fetch_file_resolves_the_path_then_downloads() -> None:
    respx.post(f'{_BASE_URL}/getFile').mock(return_value=file_response())
    download = respx.get(f'{_FILE_URL}/photos/file.jpg').mock(
        return_value=httpx.Response(200, content=b'\xff\xd8jpeg'),
    )

    content = await make_client().fetch_file('photo-file')

    assert content == b'\xff\xd8jpeg'
    assert download.called


@respx.mock
async def test_fetch_file_refuses_a_measured_file_before_downloading() -> None:
    respx.post(f'{_BASE_URL}/getFile').mock(
        return_value=file_response(file_size=6 * 1024 * 1024),
    )
    download = respx.get(f'{_FILE_URL}/photos/file.jpg').mock(
        return_value=httpx.Response(200, content=b'never read'),
    )

    with pytest.raises(MediaTooLargeError):
        await make_client().fetch_file('photo-file', limit=5 * 1024 * 1024)

    assert not download.called


@respx.mock
async def test_fetch_file_refuses_unmeasured_bytes_over_the_limit() -> None:
    respx.post(f'{_BASE_URL}/getFile').mock(
        return_value=file_response(file_size=None),
    )
    respx.get(f'{_FILE_URL}/photos/file.jpg').mock(
        return_value=httpx.Response(200, content=b'0123456789'),
    )

    with pytest.raises(MediaTooLargeError):
        await make_client().fetch_file('photo-file', limit=4)


@respx.mock
async def test_fetch_file_without_a_download_path() -> None:
    respx.post(f'{_BASE_URL}/getFile').mock(
        return_value=file_response(file_path=None),
    )

    with pytest.raises(MediaUnavailableError):
        await make_client().fetch_file('photo-file')


@respx.mock
async def test_retries_rate_limit_after_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr('app.exts.telegram.clients.asyncio.sleep', record_sleep)
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

    monkeypatch.setattr('app.exts.telegram.clients.asyncio.sleep', no_sleep)
    route = respx.post(f'{_BASE_URL}/sendMessage').mock(
        return_value=httpx.Response(503),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await make_client().send_message(123, 'hello')

    assert route.call_count == 3


@respx.mock
async def test_set_my_commands_sends_the_menu_in_the_order_it_was_given() -> None:
    route = respx.post(f'{_BASE_URL}/setMyCommands').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )

    result = await make_client().set_my_commands(
        [('help', 'List the commands.'), ('new', 'Start over.')],
    )

    # NOTE: telegram shows the menu in the order it is sent, and the app side speaks in
    # pairs so the vendor's BotCommand shape never leaves this ext
    assert json.loads(route.calls.last.request.content) == {
        'commands': [
            {'command': 'help', 'description': 'List the commands.'},
            {'command': 'new', 'description': 'Start over.'},
        ],
    }
    assert result is True
