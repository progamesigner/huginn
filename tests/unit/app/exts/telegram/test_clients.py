import json

import httpx
import pytest
import respx
from pydantic import SecretStr

from app.exts.telegram.clients import TelegramClient, webhook_url
from app.exts.telegram.exceptions import MediaTooLargeError, MediaUnavailableError
from app.exts.telegram.types import MessageEntity, Upload
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
async def test_send_message_carries_the_entities_it_was_given() -> None:
    route = respx.post(f'{_BASE_URL}/sendMessage').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': _MESSAGE}),
    )
    entities = (
        MessageEntity(type='bold', offset=0, length=5),
        MessageEntity(type='text_link', offset=6, length=5, url='https://a'),
    )

    await make_client().send_message(123, 'hello there', entities)

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': 123,
        'text': 'hello there',
        # NOTE: telegram rejects an entity carrying a field its type does not use, so the
        # optional ones only appear where they were set
        'entities': [
            {'type': 'bold', 'offset': 0, 'length': 5},
            {'type': 'text_link', 'offset': 6, 'length': 5, 'url': 'https://a'},
        ],
    }


@respx.mock
async def test_edit_message_text_carries_the_entities_it_was_given() -> None:
    route = respx.post(f'{_BASE_URL}/editMessageText').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': _MESSAGE}),
    )
    entities = (MessageEntity(type='pre', offset=0, length=7, language='python'),)

    await make_client().edit_message_text(123, 7, 'x = 1\ny', entities)

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': 123,
        'message_id': 7,
        'text': 'x = 1\ny',
        'entities': [
            {'type': 'pre', 'offset': 0, 'length': 7, 'language': 'python'},
        ],
    }


@respx.mock
async def test_send_chat_action() -> None:
    route = respx.post(f'{_BASE_URL}/sendChatAction').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )

    result = await make_client().send_chat_action(123, 'typing')

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': 123,
        'action': 'typing',
    }
    assert result is True


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
async def test_set_webhook_registers_the_url_with_the_secret_and_updates() -> None:
    route = respx.post(f'{_BASE_URL}/setWebhook').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )

    result = await make_client().set_webhook(
        'https://huginn.example.com/webhooks/telegram',
        SecretStr('webhook-secret'),
        ['message'],
        drop_pending_updates=True,
    )

    assert json.loads(route.calls.last.request.content) == {
        'url': 'https://huginn.example.com/webhooks/telegram',
        'secret_token': 'webhook-secret',
        'allowed_updates': ['message'],
        'drop_pending_updates': True,
    }
    assert result is True


@respx.mock
async def test_delete_webhook_defaults_to_keeping_pending_updates() -> None:
    route = respx.post(f'{_BASE_URL}/deleteWebhook').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )

    result = await make_client().delete_webhook()

    assert json.loads(route.calls.last.request.content) == {
        'drop_pending_updates': False,
    }
    assert result is True


@respx.mock
async def test_get_webhook_info_reads_the_registration_back() -> None:
    respx.post(f'{_BASE_URL}/getWebhookInfo').mock(
        return_value=httpx.Response(
            200,
            json={
                'ok': True,
                'result': {
                    'url': 'https://huginn.example.com/webhooks/telegram',
                    'has_custom_certificate': False,
                    'pending_update_count': 3,
                    'allowed_updates': ['message'],
                    'last_error_date': 1_721_600_000,
                    'last_error_message': 'Wrong response from the webhook: 502',
                    'max_connections': 40,
                    'ip_address': '203.0.113.7',
                },
            },
        ),
    )

    info = await make_client().get_webhook_info()

    assert info.url == 'https://huginn.example.com/webhooks/telegram'
    assert info.pending_update_count == 3
    assert info.allowed_updates == ('message',)
    assert info.last_error_date == 1_721_600_000
    assert info.max_connections == 40


@respx.mock
async def test_get_webhook_info_reads_an_unregistered_bot() -> None:
    respx.post(f'{_BASE_URL}/getWebhookInfo').mock(
        return_value=httpx.Response(
            200,
            json={
                'ok': True,
                'result': {
                    'url': '',
                    'has_custom_certificate': False,
                    'pending_update_count': 0,
                },
            },
        ),
    )

    info = await make_client().get_webhook_info()

    # NOTE: telegram reports "no webhook" as an empty url rather than an absent field
    assert not info.url
    assert info.allowed_updates == ()
    assert info.last_error_message is None


def test_webhook_url_joins_the_base_with_the_receiving_route() -> None:
    assert webhook_url('https://huginn.example.com') == (
        'https://huginn.example.com/webhooks/telegram'
    )


def test_webhook_url_tolerates_a_trailing_slash_on_the_base() -> None:
    # NOTE: pydantic's HttpUrl renders a bare origin with a trailing slash, so the
    # config-supplied base arrives in this shape
    assert webhook_url('https://huginn.example.com/') == (
        'https://huginn.example.com/webhooks/telegram'
    )


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


@respx.mock
async def test_send_file_uploads_the_bytes_as_multipart() -> None:
    route = respx.post(f'{_BASE_URL}/sendPhoto').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': _MESSAGE}),
    )

    message = await make_client().send_file(
        123,
        'photo',
        'chart.png',
        b'\x89PNG bytes',
        'image/png',
    )

    request = route.calls.last.request
    body = request.content
    assert request.headers['content-type'].startswith('multipart/form-data')
    assert b'name="chat_id"\r\n\r\n123' in body
    assert b'name="photo"; filename="chart.png"' in body
    assert b'Content-Type: image/png' in body
    assert b'\x89PNG bytes' in body
    assert message.message_id == 7


@pytest.mark.parametrize(
    ('upload', 'method'),
    [
        ('audio', 'sendAudio'),
        ('document', 'sendDocument'),
        ('photo', 'sendPhoto'),
        ('video', 'sendVideo'),
        ('voice', 'sendVoice'),
    ],
)
@respx.mock
async def test_send_file_picks_the_method_of_its_upload(
    upload: Upload,
    method: str,
) -> None:
    route = respx.post(f'{_BASE_URL}/{method}').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': _MESSAGE}),
    )

    await make_client().send_file(123, upload, 'file', b'bytes', 'text/plain')

    assert f'name="{upload}"'.encode() in route.calls.last.request.content


@respx.mock
async def test_send_file_shares_the_rate_limit_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr('app.exts.telegram.clients.asyncio.sleep', record_sleep)
    route = respx.post(f'{_BASE_URL}/sendDocument').mock(
        side_effect=[
            httpx.Response(
                429,
                json={'ok': False, 'parameters': {'retry_after': 2}},
            ),
            httpx.Response(200, json={'ok': True, 'result': _MESSAGE}),
        ],
    )

    await make_client().send_file(123, 'document', 'a.csv', b'a,b', 'text/csv')

    assert route.call_count == 2
    assert sleeps == [2.0]


@respx.mock
async def test_http_errors_do_not_name_the_bot_token() -> None:
    respx.post(f'{_BASE_URL}/sendMessage').mock(
        return_value=httpx.Response(401),
    )

    with pytest.raises(httpx.HTTPStatusError) as raised:
        await make_client().send_message(123, 'hello')

    assert raised.value.response.status_code == 401
    assert str(raised.value) == 'telegram answered 401 to sendMessage'
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__
