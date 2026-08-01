import json
from datetime import timedelta

import httpx
import pytest
import respx
from pydantic import SecretStr

from app.contracts.channels import InboundAttachment, InboundMessage
from app.exts.telegram.clients import TelegramClient
from app.exts.telegram.gateways import TelegramGateway, TelegramReply
from app.features.agents.contracts import TurnStream
from lib.http import HTTPClientManager
from tests.helpers.clock import FakeClock

_BASE_URL = 'https://api.telegram.org/bottelegram-token'
_FILE_URL = 'https://api.telegram.org/file/bottelegram-token'
_CHAT_ID = 123
_LIMIT = 4096


def make_client() -> TelegramClient:
    return TelegramClient(
        http=HTTPClientManager(),
        bot_token=SecretStr('telegram-token'),
    )


def make_reply(clock: FakeClock) -> TelegramReply:
    return TelegramReply(make_client(), _CHAT_ID, clock)


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


# NOTE: typing the parameter proves the reply contract satisfies the agent loop's sink
# NOTE: typing the parameter is the assertion -- it makes ty prove a vendor reply is
# usable as the agent loop's stream without either side importing the other
async def drive(sink: TurnStream, *fragments: str) -> None:
    for fragment in fragments:
        await sink.on_text(fragment)


@respx.mock
async def test_sends_a_draft_on_the_first_fragment() -> None:
    send = respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))

    await drive(make_reply(FakeClock()), 'He')

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
    streamer = make_reply(clock)

    await drive(streamer, 'He')
    clock.advance(timedelta(seconds=1.0))
    await drive(streamer, 'llo')

    assert not edit.called


@respx.mock
async def test_edits_once_the_throttle_window_elapses() -> None:
    respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    clock = FakeClock()
    streamer = make_reply(clock)

    await drive(streamer, 'He')
    clock.advance(timedelta(seconds=1.2))
    await drive(streamer, 'llo', ' there')

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
    streamer = make_reply(FakeClock())

    await drive(streamer, 'He', 'llo', ' there')
    await streamer.finish()

    assert texts(edit) == ['Hello there']


@respx.mock
async def test_finish_skips_the_edit_when_the_draft_is_current() -> None:
    respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_reply(FakeClock())

    await drive(streamer, 'Hello')
    await streamer.finish()

    assert not edit.called


@respx.mock
async def test_sends_nothing_when_the_turn_produced_no_text() -> None:
    send = respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_reply(FakeClock())

    await drive(streamer, '', '')
    await streamer.finish()

    assert not send.called
    assert not edit.called


@respx.mock
async def test_rolls_over_into_a_new_message_at_a_line_break() -> None:
    send = respx.post(f'{_BASE_URL}/sendMessage').mock(
        side_effect=[message(7), message(8)],
    )
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = TelegramReply(make_client(), _CHAT_ID, FakeClock(), limit=12)

    await drive(streamer, 'first line\nsecond line')
    await streamer.finish()

    assert texts(send) == ['first line', 'second line']
    assert not edit.called


@respx.mock
async def test_rollover_cuts_mid_token_when_there_is_no_whitespace() -> None:
    send = respx.post(f'{_BASE_URL}/sendMessage').mock(
        side_effect=[message(7), message(8)],
    )
    streamer = TelegramReply(make_client(), _CHAT_ID, FakeClock(), limit=5)

    await drive(streamer, 'abcdefghij')
    await streamer.finish()

    assert texts(send) == ['abcde', 'fghij']


@respx.mock
async def test_no_message_exceeds_the_telegram_length_limit() -> None:
    send = respx.post(f'{_BASE_URL}/sendMessage').mock(
        side_effect=[message(7), message(8)],
    )
    streamer = make_reply(FakeClock())

    await drive(streamer, 'word ' * 1200)
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
    streamer = TelegramReply(make_client(), _CHAT_ID, clock, limit=12)

    await drive(streamer, 'first line\nsecond')
    clock.advance(timedelta(seconds=1.2))
    await drive(streamer, ' line')

    assert json.loads(edit.calls.last.request.content) == {
        'chat_id': _CHAT_ID,
        'message_id': 8,
        'text': 'second line',
    }


@respx.mock
async def test_the_reply_milestones_are_measured_from_when_the_round_opened(
    caplog: pytest.LogCaptureFixture,
) -> None:
    respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    clock = FakeClock()
    streamer = make_reply(clock)

    with caplog.at_level('INFO'):
        clock.advance(timedelta(seconds=2))
        await drive(streamer, 'the ')
        clock.advance(timedelta(seconds=3))
        await drive(streamer, 'answer')
        await streamer.finish()

    # NOTE: the throttle timer restarts on every write, so measuring off it would report
    # the gap since the last edit instead of how long the user waited
    assert 'Sent the first draft after 2.000s' in caplog.messages
    assert 'Finished the reply after 5.000s' in caplog.messages


@respx.mock
async def test_a_rollover_does_not_report_itself_as_the_first_draft(
    caplog: pytest.LogCaptureFixture,
) -> None:
    respx.post(f'{_BASE_URL}/sendMessage').mock(side_effect=[message(7), message(8)])
    streamer = TelegramReply(make_client(), _CHAT_ID, FakeClock(), limit=12)

    with caplog.at_level('INFO'):
        await drive(streamer, 'first line\nsecond line')
        await streamer.finish()

    drafts = [
        message_
        for message_ in caplog.messages
        if message_.startswith('Sent the first')
    ]
    assert len(drafts) == 1


@respx.mock
async def test_abandon_annotates_the_draft_it_already_sent() -> None:
    respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_reply(FakeClock())

    await drive(streamer, 'Half an ans')
    await streamer.abandon()

    written = texts(edit)
    assert len(written) == 1
    assert written[0].startswith('Half an ans')
    assert 'ask again' in written[0]


@respx.mock
async def test_abandon_writes_nothing_when_no_draft_went_out() -> None:
    send = respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = respx.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))

    await make_reply(FakeClock()).abandon()

    assert not send.called
    assert not edit.called


def make_message(message_id: int = 31) -> InboundMessage:
    return InboundMessage(
        channel='support_bot',
        chat=str(_CHAT_ID),
        event='7',
        id=str(message_id),
        kind='telegram',
        text='hello',
    )


def make_gateway(clock: FakeClock) -> TelegramGateway:
    return TelegramGateway(make_client(), clock)


@respx.mock
async def test_acknowledge_reacts_to_the_incoming_message() -> None:
    route = respx.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )

    # NOTE: the contract carries the message id, so there is no longer an unparseable
    # payload for the gateway to guard against; the adapter discards those at its edge
    await make_gateway(FakeClock()).acknowledge(make_message())

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': str(_CHAT_ID),
        'message_id': '31',
        'reaction': [{'type': 'emoji', 'emoji': '\N{EYES}'}],
    }


@respx.mock
async def test_abandon_replaces_the_ack_with_the_failure_reaction() -> None:
    route = respx.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )
    gateway = make_gateway(FakeClock())

    await gateway.acknowledge(make_message())
    await gateway.abandon(make_message())

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': str(_CHAT_ID),
        'message_id': '31',
        'reaction': [
            {'type': 'emoji', 'emoji': '\N{SHOCKED FACE WITH EXPLODING HEAD}'},
        ],
    }


@respx.mock
async def test_fetch_pulls_the_bytes_behind_an_attachment_reference() -> None:
    respx.post(f'{_BASE_URL}/getFile').mock(
        return_value=httpx.Response(
            200,
            json={
                'ok': True,
                'result': {
                    'file_id': 'photo-file',
                    'file_unique_id': 'photo-unique',
                    'file_size': 5,
                    'file_path': 'photos/file.jpg',
                },
            },
        ),
    )
    respx.get(f'{_FILE_URL}/photos/file.jpg').mock(
        return_value=httpx.Response(200, content=b'\xff\xd8jpeg'),
    )
    attachment = InboundAttachment(
        kind='image',
        mime='image/jpeg',
        reference='photo-file',
        unique='photo-unique',
    )

    content = await make_gateway(FakeClock()).fetch(attachment)

    assert content == b'\xff\xd8jpeg'


@respx.mock
async def test_reply_streams_into_the_inbound_chat() -> None:
    send = respx.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    reply = make_gateway(FakeClock()).reply(make_message())

    await reply.on_text('hello')
    await reply.finish()

    assert json.loads(send.calls.last.request.content) == {
        'chat_id': str(_CHAT_ID),
        'text': 'hello',
    }
