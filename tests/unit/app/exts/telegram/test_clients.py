import json

import httpx2
import pytest
from pydantic import SecretStr

from app.exts.telegram.clients import TelegramClient, retry_after, webhook_url
from app.exts.telegram.exceptions import MediaTooLargeError, MediaUnavailableError
from app.exts.telegram.types import MessageEntity, Upload
from tests.helpers.http import MockHTTP

_BASE_URL = 'https://api.telegram.org/bottelegram-token'
_FILE_URL = 'https://api.telegram.org/file/bottelegram-token'
_MESSAGE = {
    'message_id': 7,
    'date': 1_721_600_000,
    'chat': {'id': 123, 'type': 'private'},
    'text': 'hello',
}


def make_client(http: MockHTTP) -> TelegramClient:
    return TelegramClient(
        http=http.manager(),
        bot_token=SecretStr('telegram-token'),
    )


async def test_send_message(http: MockHTTP) -> None:
    route = http.post(f'{_BASE_URL}/sendMessage').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': _MESSAGE}),
    )

    message = await make_client(http).send_message(123, 'hello')

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': 123,
        'text': 'hello',
    }
    assert message.message_id == 7


async def test_edit_message_text(http: MockHTTP) -> None:
    route = http.post(f'{_BASE_URL}/editMessageText').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': _MESSAGE}),
    )

    message = await make_client(http).edit_message_text(123, 7, 'updated')

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': 123,
        'message_id': 7,
        'text': 'updated',
    }
    assert message.message_id == 7


async def test_send_message_carries_the_entities_it_was_given(http: MockHTTP) -> None:
    route = http.post(f'{_BASE_URL}/sendMessage').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': _MESSAGE}),
    )
    entities = (
        MessageEntity(type='bold', offset=0, length=5),
        MessageEntity(type='text_link', offset=6, length=5, url='https://a'),
    )

    await make_client(http).send_message(123, 'hello there', entities)

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


async def test_edit_message_text_carries_the_entities_it_was_given(
    http: MockHTTP,
) -> None:
    route = http.post(f'{_BASE_URL}/editMessageText').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': _MESSAGE}),
    )
    entities = (MessageEntity(type='pre', offset=0, length=7, language='python'),)

    await make_client(http).edit_message_text(123, 7, 'x = 1\ny', entities)

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': 123,
        'message_id': 7,
        'text': 'x = 1\ny',
        'entities': [
            {'type': 'pre', 'offset': 0, 'length': 7, 'language': 'python'},
        ],
    }


async def test_send_chat_action(http: MockHTTP) -> None:
    route = http.post(f'{_BASE_URL}/sendChatAction').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )

    result = await make_client(http).send_chat_action(123, 'typing')

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': 123,
        'action': 'typing',
    }
    assert result is True


async def test_set_message_reaction(http: MockHTTP) -> None:
    route = http.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )

    result = await make_client(http).set_message_reaction(123, 7, '👀')

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': 123,
        'message_id': 7,
        'reaction': [{'type': 'emoji', 'emoji': '👀'}],
    }
    assert result is True


async def test_get_file(http: MockHTTP) -> None:
    route = http.post(f'{_BASE_URL}/getFile').mock(
        return_value=httpx2.Response(
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

    file = await make_client(http).get_file('voice-file')

    assert json.loads(route.calls.last.request.content) == {'file_id': 'voice-file'}
    assert file.file_path == 'voice/file.oga'


def file_response(
    *,
    file_size: int | None = 512,
    file_path: str | None = 'photos/file.jpg',
) -> httpx2.Response:
    return httpx2.Response(
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


async def test_download_file_reads_raw_bytes_from_the_file_host(http: MockHTTP) -> None:
    route = http.get(f'{_FILE_URL}/photos/file.jpg').mock(
        return_value=httpx2.Response(200, content=b'\xff\xd8jpeg'),
    )

    content = await make_client(http).download_file('photos/file.jpg')

    assert route.called
    assert content == b'\xff\xd8jpeg'


async def test_fetch_file_resolves_the_path_then_downloads(http: MockHTTP) -> None:
    http.post(f'{_BASE_URL}/getFile').mock(return_value=file_response())
    download = http.get(f'{_FILE_URL}/photos/file.jpg').mock(
        return_value=httpx2.Response(200, content=b'\xff\xd8jpeg'),
    )

    content = await make_client(http).fetch_file('photo-file')

    assert content == b'\xff\xd8jpeg'
    assert download.called


async def test_fetch_file_refuses_a_measured_file_before_downloading(
    http: MockHTTP,
) -> None:
    http.post(f'{_BASE_URL}/getFile').mock(
        return_value=file_response(file_size=6 * 1024 * 1024),
    )
    download = http.get(f'{_FILE_URL}/photos/file.jpg').mock(
        return_value=httpx2.Response(200, content=b'never read'),
    )

    with pytest.raises(MediaTooLargeError):
        await make_client(http).fetch_file('photo-file', limit=5 * 1024 * 1024)

    assert not download.called


async def test_fetch_file_refuses_unmeasured_bytes_over_the_limit(
    http: MockHTTP,
) -> None:
    http.post(f'{_BASE_URL}/getFile').mock(
        return_value=file_response(file_size=None),
    )
    http.get(f'{_FILE_URL}/photos/file.jpg').mock(
        return_value=httpx2.Response(200, content=b'0123456789'),
    )

    with pytest.raises(MediaTooLargeError):
        await make_client(http).fetch_file('photo-file', limit=4)


async def test_fetch_file_without_a_download_path(http: MockHTTP) -> None:
    http.post(f'{_BASE_URL}/getFile').mock(
        return_value=file_response(file_path=None),
    )

    with pytest.raises(MediaUnavailableError):
        await make_client(http).fetch_file('photo-file')


async def test_retries_rate_limit_after_retry_after(
    http: MockHTTP,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr('app.exts.telegram.clients.asyncio.sleep', record_sleep)
    route = http.post(f'{_BASE_URL}/sendMessage').mock(
        responses=[
            httpx2.Response(
                429,
                json={
                    'ok': False,
                    'error_code': 429,
                    'parameters': {'retry_after': 3},
                },
            ),
            httpx2.Response(200, json={'ok': True, 'result': _MESSAGE}),
        ],
    )

    await make_client(http).send_message(123, 'hello')

    assert route.call_count == 2
    assert sleeps == [3.0]


async def test_retries_server_errors_then_raises(
    http: MockHTTP,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr('app.exts.telegram.clients.asyncio.sleep', no_sleep)
    route = http.post(f'{_BASE_URL}/sendMessage').mock(
        return_value=httpx2.Response(503),
    )

    with pytest.raises(httpx2.HTTPStatusError):
        await make_client(http).send_message(123, 'hello')

    assert route.call_count == 3


async def test_set_webhook_registers_the_url_with_the_secret_and_updates(
    http: MockHTTP,
) -> None:
    route = http.post(f'{_BASE_URL}/setWebhook').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )

    result = await make_client(http).set_webhook(
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


async def test_delete_webhook_defaults_to_keeping_pending_updates(
    http: MockHTTP,
) -> None:
    route = http.post(f'{_BASE_URL}/deleteWebhook').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )

    result = await make_client(http).delete_webhook()

    assert json.loads(route.calls.last.request.content) == {
        'drop_pending_updates': False,
    }
    assert result is True


async def test_get_webhook_info_reads_the_registration_back(http: MockHTTP) -> None:
    http.post(f'{_BASE_URL}/getWebhookInfo').mock(
        return_value=httpx2.Response(
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

    info = await make_client(http).get_webhook_info()

    assert info.url == 'https://huginn.example.com/webhooks/telegram'
    assert info.pending_update_count == 3
    assert info.allowed_updates == ('message',)
    assert info.last_error_date == 1_721_600_000
    assert info.max_connections == 40


async def test_get_webhook_info_reads_an_unregistered_bot(http: MockHTTP) -> None:
    http.post(f'{_BASE_URL}/getWebhookInfo').mock(
        return_value=httpx2.Response(
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

    info = await make_client(http).get_webhook_info()

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


async def test_set_my_commands_sends_the_menu_in_the_order_it_was_given(
    http: MockHTTP,
) -> None:
    route = http.post(f'{_BASE_URL}/setMyCommands').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )

    result = await make_client(http).set_my_commands(
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


async def test_send_file_uploads_the_bytes_as_multipart(http: MockHTTP) -> None:
    route = http.post(f'{_BASE_URL}/sendPhoto').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': _MESSAGE}),
    )

    message = await make_client(http).send_file(
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
async def test_send_file_picks_the_method_of_its_upload(
    http: MockHTTP,
    upload: Upload,
    method: str,
) -> None:
    route = http.post(f'{_BASE_URL}/{method}').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': _MESSAGE}),
    )

    await make_client(http).send_file(123, upload, 'file', b'bytes', 'text/plain')

    assert f'name="{upload}"'.encode() in route.calls.last.request.content


async def test_send_file_shares_the_rate_limit_retry(
    http: MockHTTP,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr('app.exts.telegram.clients.asyncio.sleep', record_sleep)
    route = http.post(f'{_BASE_URL}/sendDocument').mock(
        responses=[
            httpx2.Response(
                429,
                json={'ok': False, 'parameters': {'retry_after': 2}},
            ),
            httpx2.Response(200, json={'ok': True, 'result': _MESSAGE}),
        ],
    )

    await make_client(http).send_file(123, 'document', 'a.csv', b'a,b', 'text/csv')

    assert route.call_count == 2
    assert sleeps == [2.0]


async def test_http_errors_do_not_name_the_bot_token(http: MockHTTP) -> None:
    http.post(f'{_BASE_URL}/sendMessage').mock(
        return_value=httpx2.Response(401),
    )

    with pytest.raises(httpx2.HTTPStatusError) as raised:
        await make_client(http).send_message(123, 'hello')

    assert raised.value.response.status_code == 401
    assert str(raised.value) == 'telegram answered 401 to sendMessage'
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


async def test_personal_menu_scope_and_language_payload(http: MockHTTP) -> None:
    scope = {'type': 'chat_member', 'chat_id': '-100', 'user_id': 77}
    set_route = http.post(f'{_BASE_URL}/setMyCommands').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )
    delete_route = http.post(f'{_BASE_URL}/deleteMyCommands').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )
    client = make_client(http)
    await client.set_my_commands(
        (('review', 'Review work'),),
        scope=scope,
        language='zh',
    )
    await client.delete_my_commands(scope=scope, language='zh')
    assert json.loads(set_route.calls.last.request.content) == {
        'commands': [{'command': 'review', 'description': 'Review work'}],
        'scope': scope,
        'language_code': 'zh',
    }
    assert json.loads(delete_route.calls.last.request.content) == {
        'scope': scope,
        'language_code': 'zh',
    }


async def test_send_message_draft_streams_a_preview(http: MockHTTP) -> None:
    route = http.post(f'{_BASE_URL}/sendMessageDraft').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )

    result = await make_client(http).send_message_draft(
        123,
        4,
        'hello',
        [MessageEntity(type='bold', offset=0, length=5)],
    )

    assert result is True
    assert json.loads(route.calls.last.request.content) == {
        'chat_id': 123,
        'draft_id': 4,
        'text': 'hello',
        'entities': [{'type': 'bold', 'offset': 0, 'length': 5}],
    }


async def test_a_single_attempt_surfaces_the_rate_limit_instead_of_waiting(
    http: MockHTTP,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(delay: float) -> None:
        raise AssertionError('a single attempt must not wait')

    monkeypatch.setattr('app.exts.telegram.clients.asyncio.sleep', no_sleep)
    route = http.post(f'{_BASE_URL}/editMessageText').mock(
        return_value=httpx2.Response(
            429,
            json={'ok': False, 'error_code': 429, 'parameters': {'retry_after': 7}},
        ),
    )

    with pytest.raises(httpx2.HTTPStatusError) as caught:
        await make_client(http).edit_message_text(123, 7, 'hello', attempts=1)

    assert route.call_count == 1
    assert retry_after(caught.value) == 7.0


def test_retry_after_ignores_errors_that_are_not_rate_limits() -> None:
    request = httpx2.Request('POST', f'{_BASE_URL}/sendMessage')
    response = httpx2.Response(400, json={'ok': False}, request=request)

    error = httpx2.HTTPStatusError('bad', request=request, response=response)

    assert retry_after(error) is None
