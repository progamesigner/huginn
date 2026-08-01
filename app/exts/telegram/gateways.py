import logging
from datetime import timedelta
from typing import override

from pydantic import SecretStr

from app.contracts.channels import (
    ChannelGatewayContract,
    ChannelReplyContract,
    InboundAttachment,
    InboundMessage,
    ToolPhase,
)
from lib.http import HTTPClientContract
from lib.temporal import ClockContract, Timer

from .clients import TelegramClient
from .types import MESSAGE

logger = logging.getLogger(__name__)

# NOTE: telegram only accepts reactions from its own allowed set, and a bot holds one
# reaction per message, so the failure replaces the ack rather than joining it
_ACK_EMOJI = '👀'
_FAIL_EMOJI = '🤯'
_FAIL_NOTE = (
    '\n\nSomething went wrong before I could finish this reply. Please ask again.'
)


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
    started: bool
    opened: Timer
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
        self.started = False
        # NOTE: `timer` is reset on every write to throttle edits; `opened` never is, so
        # the two reply milestones stay measured from the moment the round opened
        self.opened = clock.timer()
        self.timer = clock.timer()

    @override
    async def abandon(self) -> None:
        # NOTE: with no draft out there is nothing to annotate, and the reaction on the
        # incoming message already carries the whole signal
        if self.draft is None:
            return

        self.pending += _FAIL_NOTE
        await self._flush()

    @override
    async def finish(self) -> None:
        await self._flush()
        logger.info(
            'Finished the reply after %.3fs',
            self.opened.elapsed().total_seconds(),
        )

    @override
    async def on_routed(self, agent: str) -> None:
        logger.debug('Routed the turn to agent %s', agent)

    @override
    async def on_text(self, fragment: str) -> None:
        self.pending += fragment
        if self.draft is None or self.timer.elapsed() >= self.interval:
            await self._flush()

    @override
    async def on_tool(self, name: str, phase: ToolPhase) -> None:
        logger.debug('Tool %s %s', name, phase)

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
            # NOTE: a rollover clears `draft` too, so without this the continuation
            # message would report itself as the first one
            if not self.started:
                self.started = True
                logger.info(
                    'Sent the first draft after %.3fs',
                    self.opened.elapsed().total_seconds(),
                )
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
