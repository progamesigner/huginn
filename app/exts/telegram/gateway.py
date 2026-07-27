import logging
from datetime import timedelta
from typing import override

from pydantic import ValidationError

from app.ports.channels import (
    ChannelGatewayContract,
    ChannelReplyContract,
    InboundMessage,
)
from lib.temporal import ClockContract, Timer

from .client import TelegramClient
from .types import MESSAGE, Update

logger = logging.getLogger(__name__)

# NOTE: telegram only accepts reactions from its own allowed set
_ACK_EMOJI = '👀'


def _message_id(raw: bytes) -> int | None:
    try:
        update = Update.model_validate_json(raw)
    except ValidationError:
        return None

    return None if update.message is None else update.message.message_id


def _split(text: str, limit: int) -> tuple[str, str]:
    window = text[:limit]
    cut = window.rfind('\n')
    if cut <= 0:
        cut = window.rfind(' ')
    if cut <= 0:
        return window, text[limit:]

    return text[:cut], text[cut:].lstrip()


class TelegramReply(ChannelReplyContract):
    client: TelegramClient
    chat_id: int | str
    clock: ClockContract
    interval: timedelta
    limit: int
    draft: int | None
    sent: str
    pending: str
    timer: Timer

    def __init__(
        self,
        client: TelegramClient,
        chat_id: int | str,
        clock: ClockContract,
        *,
        interval: timedelta = MESSAGE.edit_interval,
        limit: int = MESSAGE.length_limit,
    ) -> None:
        self.client = client
        self.chat_id = chat_id
        self.clock = clock
        self.interval = interval
        self.limit = limit
        self.draft = None
        self.sent = ''
        self.pending = ''
        self.timer = clock.timer()

    @override
    async def finish(self) -> None:
        await self._flush()

    @override
    async def on_text(self, fragment: str) -> None:
        self.pending += fragment
        if self.draft is None or self.timer.elapsed() >= self.interval:
            await self._flush()

    async def _flush(self) -> None:
        while len(self.pending) > self.limit:
            head, tail = _split(self.pending, self.limit)
            logger.info(
                'Draft reached the %d character limit; continuing in a new message',
                self.limit,
            )
            await self._write(head)
            self.draft = None
            self.sent = ''
            self.pending = tail

        await self._write(self.pending)

    async def _write(self, text: str) -> None:
        # NOTE: telegram answers an unchanged edit with 400 message is not modified
        if not text or text == self.sent:
            return

        if self.draft is None:
            self.draft = (await self.client.send_message(self.chat_id, text)).message_id
        else:
            await self.client.edit_message_text(self.chat_id, self.draft, text)

        self.sent = text
        self.timer = self.clock.timer()


class TelegramGateway(ChannelGatewayContract):
    client: TelegramClient
    clock: ClockContract

    def __init__(self, client: TelegramClient, clock: ClockContract) -> None:
        self.client = client
        self.clock = clock

    @override
    async def acknowledge(self, message: InboundMessage) -> None:
        # NOTE: the vendor's own bytes, so the vendor's adapter reads them
        if (message_id := _message_id(message.raw)) is None:
            logger.info('Nothing to acknowledge in event %s', message.event)
            return

        await self.client.set_message_reaction(message.chat, message_id, _ACK_EMOJI)

    @override
    def reply(self, message: InboundMessage) -> ChannelReplyContract:
        return TelegramReply(self.client, message.chat, self.clock)
