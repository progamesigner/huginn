import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

from app.features.messages.streams import QueuedReply, drain
from app.features.messages.types import (
    DoneEvent,
    ErrorEvent,
    PingEvent,
    RoutedEvent,
    TextEvent,
    ToolEvent,
    TurnEvent,
)


async def answer(reply: QueuedReply) -> None:
    await reply.on_routed('scout')
    await reply.on_tool('websearch', 'started')
    await reply.on_tool('websearch', 'finished')
    await reply.on_text('the ')
    await reply.on_text('answer')
    await reply.finish()


async def collect(events: AsyncIterator[TurnEvent]) -> list[TurnEvent]:
    return [event async for event in events]


async def test_a_whole_turn_arrives_in_the_order_it_happened() -> None:
    reply = QueuedReply()

    events = await collect(drain(answer(reply), reply))

    assert events == [
        RoutedEvent(agent='scout'),
        ToolEvent(name='websearch', phase='started'),
        ToolEvent(name='websearch', phase='finished'),
        TextEvent(text='the '),
        TextEvent(text='answer'),
        DoneEvent(),
    ]


async def test_the_stream_ends_when_the_turn_returns() -> None:
    # NOTE: the sentinel comes from the turn finishing, not from finish() -- the turn
    # still has a transcript to write after the last token
    reply = QueuedReply()
    recorded = []

    async def turn() -> None:
        await reply.finish()
        recorded.append('recorded')

    events = await collect(drain(turn(), reply))

    assert events == [DoneEvent()]
    assert recorded == ['recorded']


async def test_a_failing_turn_ends_the_stream_with_an_error() -> None:
    reply = QueuedReply()

    async def turn() -> None:
        await reply.on_text('half an ')
        raise RuntimeError('boom')

    events = await collect(drain(turn(), reply))

    assert events == [
        TextEvent(text='half an '),
        ErrorEvent(message='the turn failed before it finished'),
    ]


async def test_an_idle_turn_is_kept_alive() -> None:
    reply = QueuedReply()
    release = asyncio.Event()

    async def turn() -> None:
        await release.wait()
        await reply.finish()

    events = []
    async with aclosing(drain(turn(), reply, interval=0.01)) as stream:
        async for event in stream:
            events.append(event)
            if len(events) == 2:
                release.set()

    assert events[:2] == [PingEvent(), PingEvent()]
    assert events[-1] == DoneEvent()


async def test_a_client_that_leaves_takes_the_turn_with_it() -> None:
    # NOTE: a turn nobody is listening to still spends tokens on every iteration, and a
    # reader that simply stops reading closes the stream without cancelling anything --
    # so the timeout is the assertion here: with no explicit cancel, closing the stream
    # waits out the whole turn instead of ending it
    reply = QueuedReply()
    finished = asyncio.Event()

    async def turn() -> None:
        await reply.on_text('first')
        try:
            await asyncio.sleep(30)
        finally:
            finished.set()

    async with asyncio.timeout(1):
        async with aclosing(drain(turn(), reply)) as stream:
            async for _ in stream:
                break

        assert finished.is_set()


async def test_the_queue_pauses_a_turn_the_reader_cannot_keep_up_with() -> None:
    # NOTE: bounded queue -- without it a fast model and a slow client grow the backlog
    # until the process runs out of memory
    reply = QueuedReply(pending=2)
    written = []

    async def turn() -> None:
        for index in range(5):
            await reply.on_text(str(index))
            written.append(index)
        await reply.finish()

    async with aclosing(drain(turn(), reply)) as stream:
        await asyncio.sleep(0)
        paused = list(written)
        events = [event async for event in stream]

    assert len(paused) < 5
    assert len(events) == 6


async def test_the_reply_reports_a_failure_as_an_error_event() -> None:
    reply = QueuedReply()

    await reply.abandon()

    assert reply.events.get_nowait() == ErrorEvent(
        message='the turn failed before it finished',
    )


async def test_closing_the_reply_ends_the_stream() -> None:
    reply = QueuedReply()

    await reply.close()

    assert reply.events.get_nowait() is None


async def test_a_cancelled_turn_closes_the_stream_without_a_done() -> None:
    # NOTE: cancellation means either the client left or the process is going down --
    # there is nobody left to tell, so the stream just ends
    reply = QueuedReply()

    async def turn() -> None:
        raise asyncio.CancelledError

    assert await collect(drain(turn(), reply)) == []
