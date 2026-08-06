import asyncio
import logging
from bisect import bisect_right
from collections.abc import Sequence
from datetime import timedelta
from typing import override

import httpx
from pydantic import SecretStr

from app.contracts.channels import (
    ChannelGatewayContract,
    ChannelReplyContract,
    InboundAttachment,
    InboundMessage,
    ToolPhase,
    TurnEnding,
)
from lib.http import HTTPClientContract
from lib.temporal import ClockContract, Timer

from .clients import TelegramClient
from .types import MESSAGE, TYPING, MessageEntity, clip_entities, render, utf16_offsets

logger = logging.getLogger(__name__)

# NOTE: telegram only accepts reactions from its own allowed set, and a bot holds one
# reaction per message, so the failure replaces the ack rather than joining it
_ACK_EMOJI = '👀'
_FAIL_EMOJI = '🤯'
_FAIL_NOTE = (
    '\n\nSomething went wrong before I could finish this reply. Please ask again.'
)
_ENDING_NOTES: dict[TurnEnding, str] = {
    'timeout': '\n\nI ran out of time before finishing this. Ask me to continue.',
    'exhausted': '\n\nI used up every step I get for one reply. Ask me to continue.',
}


def _split(
    text: str,
    entities: Sequence[MessageEntity],
    limit: int,
) -> list[tuple[str, tuple[MessageEntity, ...]]]:
    offsets = utf16_offsets(text)
    chunks = []
    start = 0
    while offsets[-1] - offsets[start] > limit:
        # NOTE: the floor only matters when one character is wider than the whole budget,
        # which cannot happen at the vendor's limit but would otherwise spin forever
        edge = max(bisect_right(offsets, offsets[start] + limit) - 1, start + 1)
        window = text[start:edge]
        cut = window.rfind('\n')
        if cut <= 0:
            cut = window.rfind(' ')

        head = start + cut if cut > 0 else edge
        chunks.append(
            (text[start:head], clip_entities(entities, offsets[start], offsets[head])),
        )
        # NOTE: the whitespace a soft cut lands on belongs to neither message; a hard cut
        # has none to drop, so it carries every character over
        start = len(text) - len(text[head:].lstrip()) if cut > 0 else edge

    chunks.append((text[start:], clip_entities(entities, offsets[start], offsets[-1])))
    return chunks


class TelegramReply(ChannelReplyContract):
    client: TelegramClient
    chat_id: int | str
    clock: ClockContract
    interval: timedelta
    typing_interval: timedelta
    limit: int
    draft: int | None
    sent: tuple[str, tuple[MessageEntity, ...]]
    pending: str
    sealed: int
    started: bool
    opened: Timer
    timer: Timer
    typing: asyncio.Task[None] | None

    def __init__(
        self,
        client: TelegramClient,
        chat_id: int | str,
        clock: ClockContract,
        *,
        interval: timedelta = MESSAGE.edit_interval,
        limit: int = MESSAGE.length_limit,
        typing_interval: timedelta = TYPING.refresh_interval,
    ) -> None:
        self.client = client
        self.chat_id = chat_id
        self.clock = clock
        self.interval = interval
        self.typing_interval = typing_interval
        self.limit = limit
        self.draft = None
        self.sent = ('', ())
        self.pending = ''
        self.sealed = 0
        self.started = False
        # NOTE: `timer` is reset on every write to throttle edits; `opened` never is, so
        # the two reply milestones stay measured from the moment the round opened
        self.opened = clock.timer()
        self.timer = clock.timer()
        self.typing = None

    @override
    async def abandon(self) -> None:
        await self._stop_typing()
        # NOTE: with no draft out there is nothing to annotate, and the reaction on the
        # incoming message already carries the whole signal
        if self.draft is None:
            return

        self.pending += _FAIL_NOTE
        await self._flush()

    @override
    async def finish(self) -> None:
        # NOTE: stopping before the last write
        await self._stop_typing()
        await self._flush()
        logger.info(
            'Finished the reply after %.3fs',
            self.opened.elapsed().total_seconds(),
        )

    @override
    async def on_routed(self, agent: str) -> None:
        logger.debug('Routed the turn to agent %s', agent)
        self.typing = asyncio.create_task(self._keep_typing())

    @override
    async def on_text(self, fragment: str) -> None:
        self.pending += fragment
        if self.draft is None or self.timer.elapsed() >= self.interval:
            await self._flush()

    @override
    async def on_tool(self, name: str, phase: ToolPhase) -> None:
        logger.debug('Tool %s %s', name, phase)

    @override
    async def unfinished(self, ending: TurnEnding) -> None:
        if (note := _ENDING_NOTES.get(ending)) is not None:
            self.pending += note

    async def _flush(self) -> None:
        chunks = _split(*render(self.pending), self.limit)
        for index, chunk in enumerate(chunks[:-1]):
            if index < self.sealed:
                continue

            logger.info(
                'Draft reached the %d character limit; continuing in a new message',
                self.limit,
            )
            await self._write(*chunk)
            self.draft = None
            self.sent = ('', ())
            self.sealed += 1

        await self._write(*chunks[-1])

    async def _keep_typing(self) -> None:
        while True:
            try:
                await self.client.send_chat_action(self.chat_id, TYPING.action)
            except Exception as error:
                logger.debug('Stopped showing the typing status: %s', error)
                return

            await asyncio.sleep(self.typing_interval.total_seconds())

    async def _push(self, text: str, entities: Sequence[MessageEntity]) -> None:
        if self.draft is None:
            drafted = await self.client.send_message(self.chat_id, text, entities)
            self.draft = drafted.message_id
            # NOTE: a rollover clears `draft` too, so without this the continuation
            # message would report itself as the first one
            if not self.started:
                self.started = True
                logger.info(
                    'Sent the first draft after %.3fs',
                    self.opened.elapsed().total_seconds(),
                )
        else:
            await self.client.edit_message_text(
                self.chat_id,
                self.draft,
                text,
                entities,
            )

    async def _stop_typing(self) -> None:
        if self.typing is None:
            return

        self.typing.cancel()
        await asyncio.gather(self.typing, return_exceptions=True)
        self.typing = None

    async def _write(self, text: str, entities: tuple[MessageEntity, ...] = ()) -> None:
        if not text or (text, entities) == self.sent:
            return

        try:
            await self._push(text, entities)
        except httpx.HTTPStatusError as error:
            # NOTE: telegram validates entities itself
            if not entities or error.response.status_code != 400:
                raise

            logger.warning('Telegram rejected the formatting; sending it as plain text')
            await self._push(text, ())

        self.sent = (text, entities)
        self.timer = self.clock.timer()


class TelegramGateway(ChannelGatewayContract):
    client: TelegramClient
    clock: ClockContract

    def __init__(self, client: TelegramClient, clock: ClockContract) -> None:
        self.client = client
        self.clock = clock

    @override
    async def abandon(self, message: InboundMessage) -> None:
        await self.client.set_message_reaction(message.chat, message.id, _FAIL_EMOJI)

    @override
    async def acknowledge(self, message: InboundMessage) -> None:
        await self.client.set_message_reaction(message.chat, message.id, _ACK_EMOJI)

    @override
    async def fetch(self, attachment: InboundAttachment) -> bytes:
        return await self.client.fetch_file(attachment.reference)

    @override
    def reply(self, message: InboundMessage) -> ChannelReplyContract:
        return TelegramReply(self.client, message.chat, self.clock)


def build_gateway(
    http: HTTPClientContract,
    bot_token: SecretStr,
    clock: ClockContract,
) -> ChannelGatewayContract:
    return TelegramGateway(TelegramClient(http=http, bot_token=bot_token), clock)
