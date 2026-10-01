import asyncio
from collections.abc import AsyncGenerator, Coroutine
from typing import override

from app.contracts.channels import ChannelReplyContract, ToolPhase, TurnEnding
from lib.observability import Logger

from .entities import (
    DoneEvent,
    ErrorEvent,
    PingEvent,
    RoutedEvent,
    TextEvent,
    ToolEvent,
    TurnEvent,
    UnfinishedEvent,
)

_FAILED = 'the turn failed before it finished'
_PENDING = 256
_PING_INTERVAL = 15.0


class QueuedReply(ChannelReplyContract):
    events: asyncio.Queue[TurnEvent | None]

    def __init__(self, pending: int = _PENDING) -> None:
        self.events = asyncio.Queue(pending)

    @override
    async def abandon(self) -> None:
        await self.events.put(ErrorEvent(message=_FAILED))

    @override
    async def finish(self) -> None:
        await self.events.put(DoneEvent())

    @override
    async def on_routed(self, agent: str) -> None:
        await self.events.put(RoutedEvent(agent=agent))

    @override
    async def on_text(self, fragment: str) -> None:
        await self.events.put(TextEvent(text=fragment))

    @override
    async def on_tool(self, name: str, phase: ToolPhase) -> None:
        await self.events.put(ToolEvent(name=name, phase=phase))

    @override
    async def unfinished(self, ending: TurnEnding) -> None:
        await self.events.put(UnfinishedEvent(ending=ending))

    async def close(self) -> None:
        await self.events.put(None)


async def guard(
    turn: Coroutine[None, None, None],
    reply: QueuedReply,
    *,
    logger: Logger,
) -> None:
    try:
        await turn
    except Exception:
        logger.exception('The turn failed while streaming')
        await reply.abandon()
    finally:
        await reply.close()


async def drain(
    turn: Coroutine[None, None, None],
    reply: QueuedReply,
    interval: float = _PING_INTERVAL,
    *,
    logger: Logger,
) -> AsyncGenerator[TurnEvent]:
    task = asyncio.create_task(guard(turn, reply, logger=logger))
    try:
        while True:
            try:
                event = await asyncio.wait_for(reply.events.get(), interval)
            except TimeoutError:
                yield PingEvent()
                continue

            if event is None:
                return

            yield event
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
