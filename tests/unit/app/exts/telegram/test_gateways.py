import asyncio
import json
from datetime import timedelta
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from app.contracts.channels import (
    ArtifactKind,
    InboundAttachment,
    InboundMessage,
    ReplyArtifact,
)
from app.exts.telegram.clients import TelegramClient
from app.exts.telegram.gateways import TelegramGateway, TelegramReply
from app.exts.telegram.types import UPLOAD
from app.features.agents.contracts import TurnStream
from tests.helpers.clock import FakeClock
from tests.helpers.http import MockHTTP, Route
from tests.helpers.logging import make_logger

_BASE_URL = 'https://api.telegram.org/bottelegram-token'
_FILE_URL = 'https://api.telegram.org/file/bottelegram-token'
_CHAT_ID = 123
_LIMIT = 4096
# NOTE: the refresh runs on wall-clock time inside a background task, so these tests turn
# it down instead of waiting out the vendor's interval
_TYPING_INTERVAL = timedelta(seconds=0.01)
_FAIL_TAIL = (
    '\n\nSomething went wrong before I could finish this reply. Please ask again.'
)


def make_client(http: MockHTTP) -> TelegramClient:
    return TelegramClient(
        http=http.manager(),
        bot_token=SecretStr('telegram-token'),
    )


class Sleeper:
    """Stands in for the writer's throttle wait, which only ends when a test says so."""

    clock: FakeClock
    delays: list[float]
    resting: asyncio.Event
    gate: asyncio.Event

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.delays = []
        self.resting = asyncio.Event()
        self.gate = asyncio.Event()

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        self.gate = asyncio.Event()
        self.resting.set()
        await self.gate.wait()
        self.clock.advance(timedelta(seconds=seconds))

    async def parked(self) -> float:
        async with asyncio.timeout(1):
            await self.resting.wait()
        return self.delays[-1]

    def release(self) -> None:
        self.resting.clear()
        self.gate.set()


def make_reply(
    http: MockHTTP,
    clock: FakeClock,
    sleeper: Sleeper | None = None,
    **options: Any,
) -> TelegramReply:
    return TelegramReply(
        make_client(http),
        _CHAT_ID,
        clock,
        sleep=sleeper or Sleeper(clock),
        logger=make_logger(),
        **options,
    )


def make_typing_reply(http: MockHTTP, clock: FakeClock) -> TelegramReply:
    return TelegramReply(
        make_client(http),
        _CHAT_ID,
        clock,
        typing_interval=_TYPING_INTERVAL,
        sleep=Sleeper(clock),
        logger=make_logger(),
    )


def typing_route(
    http: MockHTTP,
    calls: int = 1,
    status: int = 200,
) -> tuple[Route, asyncio.Event]:
    reached = asyncio.Event()
    seen = 0

    def answer(request: httpx2.Request) -> httpx2.Response:
        nonlocal seen
        seen += 1
        if seen >= calls:
            reached.set()

        ok = status == 200
        return httpx2.Response(status, json={'ok': ok, 'result': ok})

    route = http.post(f'{_BASE_URL}/sendChatAction').mock(side_effect=answer)
    return route, reached


async def observed(reached: asyncio.Event) -> None:
    # NOTE: the refresh is a real-time wait in a background task, so the bound is what
    # keeps a broken loop a failure instead of a hung suite
    async with asyncio.timeout(1):
        await reached.wait()


async def settled(route: Route) -> int:
    # NOTE: long enough for several refreshes, so a loop nobody stopped would show up as a
    # higher count rather than as a passing test
    await asyncio.sleep(_TYPING_INTERVAL.total_seconds() * 5)
    return route.call_count


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


def texts(route: Route) -> list[str]:
    return [json.loads(call.request.content)['text'] for call in route.calls]


def marks(route: Route) -> list[list[dict[str, object]]]:
    return [
        json.loads(call.request.content).get('entities', []) for call in route.calls
    ]


# NOTE: routes count calls without signalling them, so these yield to the writer until
# the count moves; the bound turns a writer that never writes into a failure
async def reached(route: Route, count: int) -> None:
    async with asyncio.timeout(1):
        while route.call_count < count:  # noqa: ASYNC110
            await asyncio.sleep(0)


# NOTE: the writer holds its lock until a write is fully accounted for, so waiting on
# it too keeps the test from racing the bookkeeping that follows the request
async def written(reply: TelegramReply, route: Route, count: int) -> None:
    async with asyncio.timeout(1):
        while route.call_count < count or reply.lock.locked():  # noqa: ASYNC110
            await asyncio.sleep(0)


def drafted(http: MockHTTP) -> Route:
    return http.post(f'{_BASE_URL}/sendMessageDraft').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )


def throttled(seconds: int) -> httpx2.Response:
    return httpx2.Response(
        429,
        json={'ok': False, 'error_code': 429, 'parameters': {'retry_after': seconds}},
    )


# NOTE: typing the parameter is the assertion -- it makes ty prove a vendor reply is
# usable as the agent loop's stream without either side importing the other
async def drive(sink: TurnStream, *fragments: str) -> None:
    for fragment in fragments:
        await sink.on_text(fragment)


async def test_sends_a_message_on_the_first_fragment(http: MockHTTP) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_reply(http, FakeClock())

    await drive(streamer, 'He')
    await written(streamer, send, 1)

    assert json.loads(send.calls.last.request.content) == {
        'chat_id': _CHAT_ID,
        'text': 'He',
    }
    assert not edit.called


async def test_holds_edits_inside_the_throttle_window(http: MockHTTP) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    clock = FakeClock()
    sleeper = Sleeper(clock)
    streamer = make_reply(http, clock, sleeper)

    await drive(streamer, 'He')
    await written(streamer, send, 1)
    await drive(streamer, 'llo')

    assert await sleeper.parked() == pytest.approx(1.1)
    assert not edit.called


async def test_edits_the_tail_once_the_window_elapses_with_no_further_text(
    http: MockHTTP,
) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    clock = FakeClock()
    sleeper = Sleeper(clock)
    streamer = make_reply(http, clock, sleeper)

    await drive(streamer, 'He')
    await written(streamer, send, 1)
    await drive(streamer, 'llo', ' there')
    await sleeper.parked()
    # NOTE: no fragment follows the release -- a model that pauses for a tool call must
    # still get what it already said onto the screen
    sleeper.release()
    await written(streamer, edit, 1)

    assert json.loads(edit.calls.last.request.content) == {
        'chat_id': _CHAT_ID,
        'message_id': 7,
        'text': 'Hello there',
    }
    assert edit.call_count == 1


async def test_a_slow_request_does_not_hold_up_the_stream(http: MockHTTP) -> None:
    started = asyncio.Event()
    opened = asyncio.Event()

    async def slow(request: httpx2.Request) -> httpx2.Response:
        started.set()
        await opened.wait()
        return message(7)

    send = http.post(f'{_BASE_URL}/sendMessage').mock(side_effect=slow)
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_reply(http, FakeClock())

    await drive(streamer, 'He')
    await observed(started)
    # NOTE: the bound is the assertion -- the request above never answers until after
    async with asyncio.timeout(1):
        await drive(streamer, 'llo', ' there')
    opened.set()
    await streamer.finish()

    assert texts(send) == ['He']
    assert texts(edit) == ['Hello there']


async def test_a_rate_limited_edit_waits_as_told_and_then_sends_the_latest_text(
    http: MockHTTP,
) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(
        responses=[throttled(5), message(7)],
    )
    clock = FakeClock()
    sleeper = Sleeper(clock)
    streamer = make_reply(http, clock, sleeper)

    await drive(streamer, 'He')
    await written(streamer, send, 1)
    await drive(streamer, 'llo')
    await sleeper.parked()
    sleeper.release()
    await written(streamer, edit, 1)

    assert await sleeper.parked() == pytest.approx(5.0)

    await drive(streamer, ' there')
    sleeper.release()
    await written(streamer, edit, 2)

    # NOTE: the client never retried the stale 'Hello' itself; the writer sent what was
    # current once the wait was over
    assert texts(edit) == ['Hello', 'Hello there']


async def test_a_dropped_connection_on_a_live_edit_is_retried_later(
    http: MockHTTP,
) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(
        responses=[httpx2.ConnectError('reset'), message(7)],
    )
    clock = FakeClock()
    sleeper = Sleeper(clock)
    streamer = make_reply(http, clock, sleeper)

    await drive(streamer, 'He')
    await written(streamer, send, 1)
    await drive(streamer, 'llo')
    await sleeper.parked()
    sleeper.release()
    await written(streamer, edit, 1)
    await sleeper.parked()
    sleeper.release()
    await written(streamer, edit, 2)
    await streamer.finish()

    assert texts(edit) == ['Hello', 'Hello']


async def test_a_failure_in_the_writer_fails_the_round(http: MockHTTP) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(
        return_value=httpx2.Response(403, json={'ok': False}),
    )
    streamer = make_reply(http, FakeClock())

    await drive(streamer, 'Hello')
    await written(streamer, send, 1)

    with pytest.raises(httpx2.HTTPStatusError):
        await drive(streamer, ' there')


async def test_finish_flushes_the_pending_tail(http: MockHTTP) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_reply(http, FakeClock())

    await drive(streamer, 'He')
    await written(streamer, send, 1)
    await drive(streamer, 'llo', ' there')
    await streamer.finish()

    assert texts(edit) == ['Hello there']


async def test_finish_skips_the_edit_when_the_draft_is_current(http: MockHTTP) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_reply(http, FakeClock())

    await drive(streamer, 'Hello')
    await written(streamer, send, 1)
    await streamer.finish()

    assert not edit.called


async def test_sends_nothing_when_the_turn_produced_no_text(http: MockHTTP) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_reply(http, FakeClock())

    await drive(streamer, '', '')
    await streamer.finish()

    assert not send.called
    assert not edit.called


async def test_rolls_over_into_a_new_message_at_a_line_break(http: MockHTTP) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(
        responses=[message(7), message(8)],
    )
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_reply(http, FakeClock(), limit=12)

    await drive(streamer, 'first line\nsecond line')
    await streamer.finish()

    assert texts(send) == ['first line', 'second line']
    assert not edit.called


async def test_rollover_cuts_mid_token_when_there_is_no_whitespace(
    http: MockHTTP,
) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(
        responses=[message(7), message(8)],
    )
    streamer = make_reply(http, FakeClock(), limit=5)

    await drive(streamer, 'abcdefghij')
    await streamer.finish()

    assert texts(send) == ['abcde', 'fghij']


async def test_no_message_exceeds_the_telegram_length_limit(http: MockHTTP) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(
        responses=[message(7), message(8)],
    )
    streamer = make_reply(http, FakeClock())

    await drive(streamer, 'word ' * 1200)
    await streamer.finish()

    written = texts(send)
    assert len(written) == 2
    assert all(len(text) <= _LIMIT for text in written)
    # NOTE: only the whitespace at the cut is lost, never a character of the reply
    assert ''.join(written).replace(' ', '') == 'word' * 1200


async def test_streams_formatted_text_instead_of_raw_markdown(http: MockHTTP) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    streamer = make_reply(http, FakeClock())

    await drive(streamer, '**Hello**')
    await streamer.finish()

    assert json.loads(send.calls.last.request.content) == {
        'chat_id': _CHAT_ID,
        'text': 'Hello',
        'entities': [{'type': 'bold', 'offset': 0, 'length': 5}],
    }


async def test_the_length_limit_is_measured_on_the_rendered_text(
    http: MockHTTP,
) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(
        responses=[message(7), message(8)],
    )
    # NOTE: the heading drops '# ' and gains '📌 ', and the emoji is two utf-16 code units,
    # so a round that fits the budget as markdown does not fit it as what the reader sees
    markdown = '# Heading\n\nbody'
    streamer = TelegramReply(
        make_client(http),
        _CHAT_ID,
        FakeClock(),
        limit=len(markdown),
        sleep=Sleeper(FakeClock()),
        logger=make_logger(),
    )

    await drive(streamer, markdown)
    await streamer.finish()

    assert texts(send) == ['\N{PUSHPIN} Heading\n', 'body']


async def test_a_sealed_message_is_never_rewritten_when_more_text_arrives(
    http: MockHTTP,
) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(
        responses=[message(7), message(8)],
    )
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(8))
    clock = FakeClock()
    sleeper = Sleeper(clock)
    streamer = make_reply(http, clock, sleeper, limit=12)

    # NOTE: every flush re-renders the whole round, so this is what pins the claim that a
    # re-render cannot move a boundary the reader has already been shown
    await drive(streamer, 'first line\nsecond')
    await written(streamer, send, 2)
    await drive(streamer, ' line')
    await sleeper.parked()
    sleeper.release()
    await written(streamer, edit, 1)
    await streamer.finish()

    # NOTE: the equality rather than a `not in` -- an empty list would pass that vacuously
    edited = [json.loads(call.request.content)['message_id'] for call in edit.calls]
    assert edited == [8]
    assert texts(edit) == ['second line']


async def test_an_entity_spanning_a_rollover_is_clipped_into_both_messages(
    http: MockHTTP,
) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(
        responses=[message(7), message(8)],
    )
    streamer = make_reply(http, FakeClock(), limit=5)

    await drive(streamer, '**aaa bbb**')
    await streamer.finish()

    assert texts(send) == ['aaa', 'bbb']
    assert marks(send) == [
        [{'type': 'bold', 'offset': 0, 'length': 3}],
        [{'type': 'bold', 'offset': 0, 'length': 3}],
    ]


async def test_falls_back_to_plain_text_when_telegram_rejects_the_entities(
    http: MockHTTP,
    caplog: pytest.LogCaptureFixture,
) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(
        responses=[httpx2.Response(400, json={'ok': False}), message(7)],
    )
    streamer = make_reply(http, FakeClock())

    with caplog.at_level('WARNING'):
        await drive(streamer, '**Hello**')
        await streamer.finish()

    assert texts(send) == ['Hello', 'Hello']
    assert marks(send) == [[{'type': 'bold', 'offset': 0, 'length': 5}], []]
    assert 'Telegram rejected the formatting' in caplog.text


async def test_a_rejection_that_is_not_about_the_formatting_still_fails_the_round(
    http: MockHTTP,
) -> None:
    http.post(f'{_BASE_URL}/sendMessage').mock(
        return_value=httpx2.Response(403, json={'ok': False}),
    )
    streamer = make_reply(http, FakeClock())

    await drive(streamer, '**Hello**')

    with pytest.raises(httpx2.HTTPStatusError):
        await streamer.finish()


async def test_the_reply_milestones_are_measured_from_when_the_round_opened(
    http: MockHTTP,
    caplog: pytest.LogCaptureFixture,
) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    clock = FakeClock()
    streamer = make_reply(http, clock)

    with caplog.at_level('INFO'):
        clock.advance(timedelta(seconds=2))
        await drive(streamer, 'the ')
        await written(streamer, send, 1)
        clock.advance(timedelta(seconds=3))
        await drive(streamer, 'answer')
        await streamer.finish()

    # NOTE: the throttle timer restarts on every write, so measuring off it would report
    # the gap since the last edit instead of how long the user waited
    assert 'Sent the first draft after 2.000s' in caplog.messages
    assert 'Finished the reply after 5.000s' in caplog.messages


async def test_a_rollover_does_not_report_itself_as_the_first_draft(
    http: MockHTTP,
    caplog: pytest.LogCaptureFixture,
) -> None:
    http.post(f'{_BASE_URL}/sendMessage').mock(responses=[message(7), message(8)])
    streamer = make_reply(http, FakeClock(), limit=12)

    with caplog.at_level('INFO'):
        await drive(streamer, 'first line\nsecond line')
        await streamer.finish()

    drafts = [
        message_
        for message_ in caplog.messages
        if message_.startswith('Sent the first')
    ]
    assert len(drafts) == 1


async def test_abandon_annotates_the_draft_it_already_sent(http: MockHTTP) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_reply(http, FakeClock())

    await drive(streamer, 'Half an ans')
    await written(streamer, send, 1)
    await streamer.abandon()

    written_ = texts(edit)
    assert len(written_) == 1
    assert written_[0].startswith('Half an ans')
    assert 'ask again' in written_[0]


async def test_abandon_still_annotates_after_the_writer_failed(http: MockHTTP) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(
        responses=[httpx2.Response(403, json={'ok': False}), message(7)],
    )
    clock = FakeClock()
    sleeper = Sleeper(clock)
    streamer = make_reply(http, clock, sleeper)

    await drive(streamer, 'Half')
    await written(streamer, send, 1)
    await drive(streamer, ' an ans')
    await sleeper.parked()
    sleeper.release()
    await written(streamer, edit, 1)
    await streamer.abandon()

    assert texts(edit)[-1] == f'Half an ans{_FAIL_TAIL}'


async def test_an_unfinished_turn_is_annotated_with_why_it_stopped(
    http: MockHTTP,
) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_reply(http, FakeClock())

    await drive(streamer, 'Half an ans')
    await written(streamer, send, 1)
    await streamer.unfinished('timeout')
    await streamer.finish()

    written_ = texts(edit)
    assert written_[-1].startswith('Half an ans')
    assert 'ran out of time' in written_[-1]


async def test_an_ending_with_nothing_said_yet_is_still_worth_a_message(
    http: MockHTTP,
) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    streamer = make_reply(http, FakeClock())

    await streamer.unfinished('exhausted')
    await streamer.finish()

    # NOTE: silence would leave the reader with an acknowledgement and no answer at all
    assert 'used up every step' in texts(send)[0]


async def test_a_completed_turn_is_never_annotated(http: MockHTTP) -> None:
    http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    streamer = make_reply(http, FakeClock())

    await drive(streamer, 'A whole answer')
    await streamer.unfinished('complete')
    await streamer.finish()

    assert not edit.called


async def test_abandon_writes_nothing_when_no_draft_went_out(http: MockHTTP) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))

    await make_reply(http, FakeClock()).abandon()

    assert not send.called
    assert not edit.called


async def test_routing_the_turn_keeps_the_typing_status_up(http: MockHTTP) -> None:
    # NOTE: waiting for a second call is the assertion -- telegram expires the status
    # after a few seconds, so one call at the top of a slow turn leaves the chat idle
    action, refreshed = typing_route(http, calls=2)
    streamer = make_typing_reply(http, FakeClock())

    await streamer.on_routed('scout')
    await observed(refreshed)
    await streamer.finish()

    assert json.loads(action.calls.last.request.content) == {
        'chat_id': _CHAT_ID,
        'action': 'typing',
    }


async def test_finish_stops_the_typing_status(http: MockHTTP) -> None:
    action, shown = typing_route(http)
    http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    streamer = make_typing_reply(http, FakeClock())

    await streamer.on_routed('scout')
    await observed(shown)
    await drive(streamer, 'the answer')
    await streamer.finish()
    stopped = action.call_count

    assert await settled(action) == stopped


async def test_abandon_stops_the_typing_status_with_no_draft_out(
    http: MockHTTP,
) -> None:
    action, shown = typing_route(http)
    streamer = make_typing_reply(http, FakeClock())

    await streamer.on_routed('scout')
    await observed(shown)
    await streamer.abandon()
    stopped = action.call_count

    assert await settled(action) == stopped


async def test_a_reply_that_never_routed_shows_no_typing_status(http: MockHTTP) -> None:
    action, _ = typing_route(http)
    http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    streamer = make_typing_reply(http, FakeClock())

    # NOTE: the shape of a slash command answer and of the unreadable-round decline --
    # neither reaches the model, so neither has a wait worth announcing
    await drive(streamer, 'here you go')
    await streamer.finish()

    assert await settled(action) == 0


async def test_a_failed_typing_call_ends_the_status_for_the_round(
    http: MockHTTP,
) -> None:
    action, tried = typing_route(http, status=400)
    streamer = make_typing_reply(http, FakeClock())

    await streamer.on_routed('scout')
    await observed(tried)
    count = await settled(action)
    await streamer.finish()

    assert count == 1


def make_message(message_id: int = 31, *, private: bool = False) -> InboundMessage:
    return InboundMessage(
        channel='support_bot',
        chat=str(_CHAT_ID),
        event='7',
        id=str(message_id),
        kind='telegram',
        text='hello',
        private=private,
    )


def make_gateway(http: MockHTTP, clock: FakeClock) -> TelegramGateway:
    return TelegramGateway(make_client(http), clock, logger=make_logger())


async def test_acknowledge_reacts_to_the_incoming_message(http: MockHTTP) -> None:
    route = http.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )

    # NOTE: the contract carries the message id, so there is no longer an unparseable
    # payload for the gateway to guard against; the adapter discards those at its edge
    await make_gateway(http, FakeClock()).acknowledge(make_message())

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': str(_CHAT_ID),
        'message_id': '31',
        'reaction': [{'type': 'emoji', 'emoji': '\N{EYES}'}],
    }


async def test_abandon_replaces_the_ack_with_the_failure_reaction(
    http: MockHTTP,
) -> None:
    route = http.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )
    gateway = make_gateway(http, FakeClock())

    await gateway.acknowledge(make_message())
    await gateway.abandon(make_message())

    assert json.loads(route.calls.last.request.content) == {
        'chat_id': str(_CHAT_ID),
        'message_id': '31',
        'reaction': [
            {'type': 'emoji', 'emoji': '\N{SHOCKED FACE WITH EXPLODING HEAD}'},
        ],
    }


async def test_fetch_pulls_the_bytes_behind_an_attachment_reference(
    http: MockHTTP,
) -> None:
    http.post(f'{_BASE_URL}/getFile').mock(
        return_value=httpx2.Response(
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
    http.get(f'{_FILE_URL}/photos/file.jpg').mock(
        return_value=httpx2.Response(200, content=b'\xff\xd8jpeg'),
    )
    attachment = InboundAttachment(
        kind='image',
        mime='image/jpeg',
        reference='photo-file',
        unique='photo-unique',
    )

    content = await make_gateway(http, FakeClock()).fetch(attachment)

    assert content == b'\xff\xd8jpeg'


async def test_reply_streams_into_the_inbound_chat(http: MockHTTP) -> None:
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    reply = make_gateway(http, FakeClock()).reply(make_message())

    await reply.on_text('hello')
    await reply.finish()

    assert json.loads(send.calls.last.request.content) == {
        'chat_id': str(_CHAT_ID),
        'text': 'hello',
    }


def artifact(
    kind: ArtifactKind = 'image',
    mime: str = 'image/png',
    *,
    size: int | None = None,
) -> ReplyArtifact:
    data = b'\x89PNG bytes'
    return ReplyArtifact(
        blob='0190a1b2-0000-7000-8000-000000000001',
        kind=kind,
        mime=mime,
        size=len(data) if size is None else size,
        data=data,
        name='chart.png',
    )


class Wire:
    calls: list[tuple[str, str]]
    status: int

    def __init__(self, status: int = 200) -> None:
        self.calls = []
        self.status = status

    def answer(self, request: httpx2.Request) -> httpx2.Response:
        method = request.url.path.rpartition('/')[2]
        if method == 'sendMessageDraft':
            self.calls.append((method, json.loads(request.content)['text']))
            return httpx2.Response(200, json={'ok': True, 'result': True})
        if method in ('sendMessage', 'editMessageText'):
            self.calls.append((method, json.loads(request.content)['text']))
            return message(7 + len(self.calls))

        self.calls.append((method, ''))
        if self.status != 200:
            return httpx2.Response(self.status, json={'ok': False})
        return message(100)


def wired(http: MockHTTP, status: int = 200) -> Wire:
    wire = Wire(status)
    http.post(url__startswith=_BASE_URL).mock(side_effect=wire.answer)
    return wire


async def test_text_before_a_file_is_finished_before_the_file_goes_out(
    http: MockHTTP,
) -> None:
    wire = wired(http)
    reply = make_reply(http, FakeClock())

    await reply.on_text('Here')
    await reply.on_text(' it is')
    await reply.on_artifact(artifact())
    await reply.on_text('Anything else?')
    await reply.finish()

    assert wire.calls == [
        ('sendMessage', 'Here it is'),
        ('sendPhoto', ''),
        ('sendMessage', 'Anything else?'),
    ]


async def test_a_file_with_no_text_before_it_goes_out_alone(http: MockHTTP) -> None:
    wire = wired(http)
    reply = make_reply(http, FakeClock())

    await reply.on_artifact(artifact())
    await reply.finish()

    assert wire.calls == [('sendPhoto', '')]


@pytest.mark.parametrize(
    ('kind', 'mime', 'method'),
    [
        ('audio', 'audio/ogg', 'sendVoice'),
        ('audio', 'audio/mpeg', 'sendAudio'),
        ('document', 'text/csv', 'sendDocument'),
        ('video', 'video/mp4', 'sendVideo'),
    ],
)
async def test_a_file_goes_out_by_the_method_its_mime_calls_for(
    http: MockHTTP,
    kind: ArtifactKind,
    mime: str,
    method: str,
) -> None:
    wire = wired(http)

    await make_reply(http, FakeClock()).on_artifact(artifact(kind, mime))

    assert wire.calls == [(method, '')]


async def test_a_file_telegram_refuses_is_explained_and_the_reply_finishes(
    http: MockHTTP,
) -> None:
    wire = wired(http, status=400)
    reply = make_reply(http, FakeClock())

    await reply.on_text('Here it is')
    await reply.on_artifact(artifact())
    await reply.finish()

    assert wire.calls == [
        ('sendMessage', 'Here it is'),
        ('sendPhoto', ''),
        (
            'sendMessage',
            'I made chart.png but could not send it (telegram answered 400).',
        ),
    ]


async def test_a_file_over_the_vendor_limit_is_never_uploaded(http: MockHTTP) -> None:
    wire = wired(http)
    reply = make_reply(http, FakeClock())

    await reply.on_artifact(artifact(size=UPLOAD.photo_limit + 1))
    await reply.finish()

    assert wire.calls == [
        ('sendMessage', 'I made chart.png but could not send it (it is too large).'),
    ]


async def test_reply_streams_a_private_chat_through_draft_previews(
    http: MockHTTP,
) -> None:
    draft = drafted(http)
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    reply = make_gateway(http, FakeClock()).reply(make_message(private=True))

    await reply.on_text('hello')
    await reached(draft, 1)
    await reply.finish()

    assert texts(draft) == ['hello']
    assert texts(send) == ['hello']


async def test_a_private_chat_streams_into_a_draft_preview(http: MockHTTP) -> None:
    draft = drafted(http)
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    streamer = make_reply(http, FakeClock(), private=True)

    await drive(streamer, 'He')
    await written(streamer, draft, 1)

    assert json.loads(draft.calls.last.request.content) == {
        'chat_id': _CHAT_ID,
        'draft_id': 1,
        'text': 'He',
    }
    assert not send.called


async def test_a_private_chat_keeps_the_finished_reply_as_a_message(
    http: MockHTTP,
) -> None:
    draft = drafted(http)
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    clock = FakeClock()
    sleeper = Sleeper(clock)
    streamer = make_reply(http, clock, sleeper, private=True)

    await drive(streamer, '**He**')
    await written(streamer, draft, 1)
    await drive(streamer, 'llo')
    await sleeper.parked()
    sleeper.release()
    await written(streamer, draft, 2)
    await streamer.finish()

    # NOTE: the draft vanishes on its own, so the same text has to land as a real message
    assert texts(draft) == ['He', 'Hello']
    assert json.loads(send.calls.last.request.content) == {
        'chat_id': _CHAT_ID,
        'text': 'Hello',
        'entities': [{'type': 'bold', 'offset': 0, 'length': 2}],
    }
    assert not edit.called


async def test_a_private_reply_seals_each_rolled_over_chunk_as_a_message(
    http: MockHTTP,
) -> None:
    draft = drafted(http)
    send = http.post(f'{_BASE_URL}/sendMessage').mock(
        responses=[message(7), message(8)],
    )
    clock = FakeClock()
    sleeper = Sleeper(clock)
    streamer = make_reply(http, clock, sleeper, private=True, limit=12)

    await drive(streamer, 'first line')
    await written(streamer, draft, 1)
    await drive(streamer, '\nsecond line')
    await sleeper.parked()
    sleeper.release()
    await written(streamer, draft, 2)
    await streamer.finish()

    assert texts(send) == ['first line', 'second line']
    # NOTE: a fresh id per chunk, so the continuation does not animate out of a preview
    # whose text already went out as the message above it
    assert [json.loads(call.request.content)['draft_id'] for call in draft.calls] == [
        1,
        2,
    ]
    assert texts(draft) == ['first line', 'second line']


async def test_a_draft_preview_is_refreshed_before_telegram_drops_it(
    http: MockHTTP,
) -> None:
    draft = drafted(http)
    http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    clock = FakeClock()
    sleeper = Sleeper(clock)
    streamer = make_reply(
        http,
        clock,
        sleeper,
        private=True,
        draft_refresh=timedelta(seconds=0.01),
    )

    await drive(streamer, 'Hi')
    await written(streamer, draft, 1)
    await sleeper.parked()
    sleeper.release()
    await written(streamer, draft, 2)
    await streamer.finish()

    assert texts(draft) == ['Hi', 'Hi']


async def test_a_refused_draft_falls_back_to_editing_a_message(
    http: MockHTTP,
    caplog: pytest.LogCaptureFixture,
) -> None:
    draft = http.post(f'{_BASE_URL}/sendMessageDraft').mock(
        return_value=httpx2.Response(400, json={'ok': False}),
    )
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    edit = http.post(f'{_BASE_URL}/editMessageText').mock(return_value=message(7))
    clock = FakeClock()
    sleeper = Sleeper(clock)
    streamer = make_reply(http, clock, sleeper, private=True)

    with caplog.at_level('WARNING'):
        await drive(streamer, 'Hello')
        await written(streamer, send, 1)
        await drive(streamer, ' there')
        await sleeper.parked()
        sleeper.release()
        await written(streamer, edit, 1)
        await streamer.finish()

    assert draft.call_count == 1
    assert texts(send) == ['Hello']
    assert texts(edit) == ['Hello there']
    assert 'Telegram refused the draft preview' in caplog.text


async def test_a_rate_limited_draft_waits_instead_of_dropping_drafts(
    http: MockHTTP,
) -> None:
    draft = http.post(f'{_BASE_URL}/sendMessageDraft').mock(
        responses=[
            throttled(3),
            httpx2.Response(200, json={'ok': True, 'result': True}),
        ],
    )
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    clock = FakeClock()
    sleeper = Sleeper(clock)
    streamer = make_reply(http, clock, sleeper, private=True)

    await drive(streamer, 'He')
    await reached(draft, 1)
    assert await sleeper.parked() == pytest.approx(3.0)
    await drive(streamer, 'llo')
    sleeper.release()
    await written(streamer, draft, 2)

    assert texts(draft) == ['He', 'Hello']
    assert not send.called


async def test_an_abandoned_private_reply_keeps_what_it_previewed(
    http: MockHTTP,
) -> None:
    draft = drafted(http)
    send = http.post(f'{_BASE_URL}/sendMessage').mock(return_value=message(7))
    streamer = make_reply(http, FakeClock(), private=True)

    await drive(streamer, 'Half an ans')
    await written(streamer, draft, 1)
    await streamer.abandon()

    (kept,) = texts(send)
    assert kept.startswith('Half an ans')
    assert 'ask again' in kept


async def test_a_private_reply_sends_its_text_as_a_message_before_a_file(
    http: MockHTTP,
) -> None:
    wire = wired(http)
    reply = make_reply(http, FakeClock(), private=True)

    await reply.on_text('Here')
    await reply.on_artifact(artifact())
    await reply.on_text('Anything else?')
    await reply.finish()

    assert wire.calls == [
        ('sendMessage', 'Here'),
        ('sendPhoto', ''),
        ('sendMessage', 'Anything else?'),
    ]
