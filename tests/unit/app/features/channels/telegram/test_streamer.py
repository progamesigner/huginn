import json
from datetime import timedelta

import httpx
import respx
from pydantic import SecretStr

from app.exts.telegram.client import TelegramClient
from app.features.agents.contracts import TextSink
from app.features.channels.telegram.streamer import TelegramStreamer
from lib.http import HTTPClientManager
from tests.helpers.clock import FakeClock

_BASE_URL = 'https://api.telegram.org/bottelegram-token'
_CHAT_ID = 123
_LIMIT = 4096


def make_client() -> TelegramClient:
    return TelegramClient(
        http=HTTPClientManager(),
        bot_token=SecretStr('telegram-token'),
    )


def make_streamer(clock: FakeClock) -> TelegramStreamer:
    return TelegramStreamer(make_client(), _CHAT_ID, clock)


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


def texts(route: respx.Route) -> list[str]:
    return [json.loads(call.request.content)['text'] for call in route.calls]


# NOTE: typing the parameter proves the streamer satisfies the agent loop's sink
async def drive(sink: TextSink, *fragments: str) -> None:
    for fragment in fragments:
        await sink(fragment)


@respx.mock
async def test_sends_a_draft_on_the_first_fragment() -> None:
    send = respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))

    await drive(make_streamer(FakeClock()).on_text, 'He')

    assert json.loads(send.calls.last.request.content) == {
        'chat_id': _CHAT_ID,
        'text': 'He',
    }
    assert not edit.called


@respx.mock
async def test_holds_edits_inside_the_throttle_window() -> None:
    respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    clock = FakeClock()
    streamer = make_streamer(clock)

    await drive(streamer.on_text, 'He')
    clock.advance(timedelta(seconds=1.0))
    await drive(streamer.on_text, 'llo')

    assert not edit.called


@respx.mock
async def test_edits_once_the_throttle_window_elapses() -> None:
    respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    clock = FakeClock()
    streamer = make_streamer(clock)

    await drive(streamer.on_text, 'He')
    clock.advance(timedelta(seconds=1.2))
    await drive(streamer.on_text, 'llo', ' there')

    assert json.loads(edit.calls.last.request.content) == {
        'chat_id': _CHAT_ID,
        'message_id': 7,
        'text': 'Hello',
    }
    assert edit.call_count == 1


@respx.mock
async def test_finish_flushes_the_pending_tail() -> None:
    respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_streamer(FakeClock())

    await drive(streamer.on_text, 'He', 'llo', ' there')
    await streamer.finish()

    assert texts(edit) == ['Hello there']


@respx.mock
async def test_finish_skips_the_edit_when_the_draft_is_current() -> None:
    respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_streamer(FakeClock())

    await drive(streamer.on_text, 'Hello')
    await streamer.finish()

    assert not edit.called


@respx.mock
async def test_sends_nothing_when_the_turn_produced_no_text() -> None:
    send = respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_streamer(FakeClock())

    await drive(streamer.on_text, '', '')
    await streamer.finish()

    assert not send.called
    assert not edit.called


@respx.mock
async def test_rolls_over_into_a_new_message_at_a_line_break() -> None:
    send = respx.post(f'{_BASE_URL}/sendMessage').mock(
        side_effect=[message(7), message(8)],
    )
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = TelegramStreamer(make_client(), _CHAT_ID, FakeClock(), limit=12)

    await drive(streamer.on_text, 'first line\nsecond line')
    await streamer.finish()

    assert texts(send) == ['first line', 'second line']
    assert not edit.called


@respx.mock
async def test_rollover_cuts_mid_token_when_there_is_no_whitespace() -> None:
    send = respx.post(f'{_BASE_URL}/sendMessage').mock(
        side_effect=[message(7), message(8)],
    )
    streamer = TelegramStreamer(make_client(), _CHAT_ID, FakeClock(), limit=5)

    await drive(streamer.on_text, 'abcdefghij')
    await streamer.finish()

    assert texts(send) == ['abcde', 'fghij']


@respx.mock
async def test_no_message_exceeds_the_telegram_length_limit() -> None:
    send = respx.post(f'{_BASE_URL}/sendMessage').mock(
        side_effect=[message(7), message(8)],
    )
    streamer = make_streamer(FakeClock())

    await drive(streamer.on_text, 'word ' * 1200)
    await streamer.finish()

    written = texts(send)
    assert len(written) == 2
    assert all(len(text) <= _LIMIT for text in written)
    # NOTE: only the whitespace at the cut is lost, never a character of the reply
    assert ''.join(written).replace(' ', '') == 'word' * 1200


@respx.mock
async def test_keeps_editing_the_continuation_message_after_a_rollover() -> None:
    respx.post(f'{_BASE_URL}/sendMessage').mock(side_effect=[message(7), message(8)])
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(8))
    clock = FakeClock()
    streamer = TelegramStreamer(make_client(), _CHAT_ID, clock, limit=12)

    await drive(streamer.on_text, 'first line\nsecond')
    clock.advance(timedelta(seconds=1.2))
    await drive(streamer.on_text, ' line')

    assert json.loads(edit.calls.last.request.content) == {
        'chat_id': _CHAT_ID,
        'message_id': 8,
        'text': 'second line',
    }
