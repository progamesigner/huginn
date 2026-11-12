import asyncio
from bisect import bisect_right
from collections.abc import Awaitable, Callable, Sequence
from datetime import timedelta
from typing import Literal, override

import httpx2
from pydantic import JsonValue, SecretStr

from app.contracts.channels import (
    ATTACHMENT_LIMITS,
    ChannelGatewayContract,
    ChannelGatewayFactoryContract,
    ChannelMenuContract,
    ChannelReplyContract,
    InboundAttachment,
    InboundMessage,
    ReplyArtifact,
    ToolPhase,
    TurnEnding,
)
from lib.http import HTTPClientContract
from lib.observability import Logger
from lib.temporal import ClockContract, Timer

from .clients import MAX_ATTEMPTS, TelegramClient, retry_after
from .types import (
    MESSAGE,
    TYPING,
    MessageEntity,
    clip_entities,
    render,
    select_upload,
    upload_limit,
    utf16_offsets,
)

# NOTE: telegram only accepts reactions from its own allowed set, and a bot holds one
# reaction per message, so the failure replaces the ack rather than joining it
_ACK_EMOJI = '👀'
_FAIL_EMOJI = '🤯'
_FAIL_NOTE = (
    '\n\nSomething went wrong before I could finish this reply. Please ask again.'
)
_UNSENT_NOTE = 'I made {name} but could not send it ({reason}).\n\n'
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


type Sleep = Callable[[float], Awaitable[None]]
type Surface = Literal['draft', 'message']


class _Throttled(Exception):
    delay: timedelta

    def __init__(self, delay: timedelta) -> None:
        super().__init__(f'telegram asked to wait {delay.total_seconds():.3f}s')
        self.delay = delay


class TelegramReply(ChannelReplyContract):
    logger: Logger
    client: TelegramClient
    chat_id: int | str
    clock: ClockContract
    interval: timedelta
    typing_interval: timedelta
    draft_refresh: timedelta
    limit: int
    sleep: Sleep
    drafting: bool
    draft_id: int
    showing: bool
    message: int | None
    sent: tuple[Surface, str, tuple[MessageEntity, ...]] | None
    pending: str
    sealed: int
    started: bool
    gap: timedelta
    opened: Timer
    timer: Timer
    dirty: asyncio.Event
    lock: asyncio.Lock
    writer: asyncio.Task[None] | None
    typing: asyncio.Task[None] | None

    def __init__(
        self,
        client: TelegramClient,
        chat_id: int | str,
        clock: ClockContract,
        *,
        private: bool = False,
        interval: timedelta = MESSAGE.edit_interval,
        limit: int = MESSAGE.length_limit,
        typing_interval: timedelta = TYPING.refresh_interval,
        draft_refresh: timedelta = MESSAGE.draft_refresh,
        sleep: Sleep = asyncio.sleep,
        logger: Logger,
    ) -> None:
        self.logger = logger.bind(component='telegram')
        self.client = client
        self.chat_id = chat_id
        self.clock = clock
        self.interval = interval
        self.typing_interval = typing_interval
        self.draft_refresh = draft_refresh
        self.limit = limit
        self.sleep = sleep
        # NOTE: telegram only shows draft previews in private chats; elsewhere the reply
        # streams by editing a message it already sent
        self.drafting = private
        self.draft_id = 1
        self.showing = False
        self.message = None
        self.sent = None
        self.pending = ''
        self.sealed = 0
        self.started = False
        # NOTE: the first write goes out at once; every later one waits out `gap` after
        # the write before it, which is the edit interval or what a 429 asked for
        self.gap = timedelta(0)
        # NOTE: `timer` is reset on every write to throttle edits; `opened` never is, so
        # the two reply milestones stay measured from the moment the round opened
        self.opened = clock.timer()
        self.timer = clock.timer()
        self.dirty = asyncio.Event()
        self.lock = asyncio.Lock()
        self.writer = None
        self.typing = None

    @override
    async def abandon(self) -> None:
        await self._stop_typing()
        try:
            await self._stop_writer()
        except Exception:
            self.logger.warning(
                'Streaming the reply failed before the round was abandoned',
                exc_info=True,
            )

        # NOTE: with nothing out or owed there is nothing to annotate, and the reaction
        # on the incoming message already carries the whole signal
        if not self.pending:
            return

        self.pending += _FAIL_NOTE
        await self._flush(final=True)

    @override
    async def finish(self) -> None:
        # NOTE: stopping before the last write
        await self._stop_typing()
        await self._stop_writer()
        await self._flush(final=True)
        self.logger.info(
            'Finished the reply after %.3fs',
            self.opened.elapsed().total_seconds(),
        )

    # NOTE: the text so far is finished before the file goes out, so the file lands
    # after it and the next text opens a message of its own instead of editing above it
    @override
    async def on_artifact(self, artifact: ReplyArtifact) -> None:
        await self._stop_writer()
        await self._flush(final=True)
        self.pending = ''
        self.sealed = 0

        name = artifact.name or f'{artifact.kind}'
        upload = select_upload(artifact.kind, artifact.mime)
        if artifact.size > (limit := upload_limit(upload)):
            self.logger.warning(
                'Not sending %s: %d bytes is over the %d byte limit',
                artifact.blob,
                artifact.size,
                limit,
            )
            self.pending += _UNSENT_NOTE.format(name=name, reason='it is too large')
            return

        try:
            await self.client.send_file(
                self.chat_id,
                upload,
                name,
                artifact.data,
                artifact.mime,
            )
        except Exception as error:
            self.logger.exception('Could not send %s as a %s', artifact.blob, upload)
            reason = (
                f'telegram answered {error.response.status_code}'
                if isinstance(error, httpx2.HTTPStatusError)
                else 'the upload failed'
            )
            self.pending += _UNSENT_NOTE.format(name=name, reason=reason)

    @override
    async def on_routed(self, agent: str) -> None:
        self.logger.debug('Routed the turn to agent %s', agent)
        self.typing = asyncio.create_task(self._keep_typing())

    # NOTE: only queues the text; the writer task talks to telegram, so a slow or
    # throttled request never holds up the model stream that feeds this
    @override
    async def on_text(self, fragment: str) -> None:
        if self.writer is not None and self.writer.done():
            self.writer.result()

        self.pending += fragment
        if self.writer is None:
            self.writer = asyncio.create_task(self._write_loop())
        self.dirty.set()

    @override
    async def on_tool(self, name: str, phase: ToolPhase) -> None:
        self.logger.debug('Tool %s %s', name, phase)

    @override
    async def unfinished(self, ending: TurnEnding) -> None:
        if (note := _ENDING_NOTES.get(ending)) is not None:
            self.pending += note

    async def _flush(self, *, final: bool) -> None:
        chunks = _split(*render(self.pending), self.limit)
        for index, chunk in enumerate(chunks[:-1]):
            if index < self.sealed:
                continue

            self.logger.info(
                'Draft reached the %d character limit; continuing in a new message',
                self.limit,
            )
            await self._seal(*chunk)
            self.sealed += 1

        if final:
            await self._seal(*chunks[-1])
        else:
            await self._write(*chunks[-1], live=True)

    async def _idle(self) -> None:
        if not (self.drafting and self.showing):
            await self.dirty.wait()
            return

        try:
            async with asyncio.timeout(self.draft_refresh.total_seconds()):
                await self.dirty.wait()
        except TimeoutError:
            # NOTE: a long tool call would otherwise outlast the preview, leaving the
            # reader with nothing on screen until the final message lands
            self.sent = None

    async def _keep_typing(self) -> None:
        while True:
            try:
                await self.client.send_chat_action(self.chat_id, TYPING.action)
            except Exception as error:
                self.logger.debug(
                    'Stopped showing the typing status: %s',
                    error,
                    exc_info=True,
                )
                return

            await asyncio.sleep(self.typing_interval.total_seconds())

    async def _push(
        self,
        surface: Surface,
        text: str,
        entities: Sequence[MessageEntity],
        attempts: int,
    ) -> None:
        if surface == 'draft':
            await self.client.send_message_draft(
                self.chat_id,
                self.draft_id,
                text,
                entities,
                attempts=attempts,
            )
            self.showing = True
        elif self.message is None:
            sent = await self.client.send_message(
                self.chat_id,
                text,
                entities,
                attempts=attempts,
            )
            self.message = sent.message_id
        else:
            await self.client.edit_message_text(
                self.chat_id,
                self.message,
                text,
                entities,
                attempts=attempts,
            )

        # NOTE: a rollover clears `message` too, so a flag rather than the message id
        # keeps the continuation from reporting itself as the first one
        if not self.started:
            self.started = True
            self.logger.info(
                'Sent the first draft after %.3fs',
                self.opened.elapsed().total_seconds(),
            )

    # NOTE: a chunk is done once more text spilled past it or the round is over; in a
    # private chat this is the only point it becomes a real message
    async def _seal(self, text: str, entities: tuple[MessageEntity, ...]) -> None:
        await self._write(text, entities, live=False)
        self.message = None
        self.sent = None
        if self.showing:
            self.draft_id += 1
            self.showing = False

    async def _stop_typing(self) -> None:
        if self.typing is None:
            return

        self.typing.cancel()
        await asyncio.gather(self.typing, return_exceptions=True)
        self.typing = None

    async def _stop_writer(self) -> None:
        writer, self.writer = self.writer, None
        if writer is None:
            return

        # NOTE: taking the lock first lets an in-flight write land, so nothing the
        # writer sent can arrive after the write the caller is about to make
        async with self.lock:
            writer.cancel()
        (outcome,) = await asyncio.gather(writer, return_exceptions=True)
        self.dirty.clear()
        if isinstance(outcome, Exception):
            raise outcome

    async def _write(
        self,
        text: str,
        entities: tuple[MessageEntity, ...] = (),
        *,
        live: bool,
    ) -> None:
        surface = 'draft' if live and self.drafting else 'message'
        if not text or (surface, text, entities) == self.sent:
            return

        # NOTE: a live write is superseded by the next one anyway, so it takes a single
        # attempt and leaves the waiting to the writer instead of the client
        attempts = 1 if live else MAX_ATTEMPTS
        try:
            await self._push(surface, text, entities, attempts)
        except httpx2.TransportError as error:
            if not live:
                raise
            raise _Throttled(self.interval) from error
        except httpx2.HTTPStatusError as error:
            status = error.response.status_code
            if live and (delay := retry_after(error)) is not None:
                raise _Throttled(timedelta(seconds=delay)) from error
            if live and status >= 500:
                raise _Throttled(self.interval) from error
            if surface == 'draft':
                self.logger.warning(
                    'Telegram refused the draft preview (%d); editing a message instead',
                    status,
                )
                self.drafting = False
                self.showing = False
                await self._write(text, entities, live=live)
                return
            # NOTE: telegram validates entities itself
            if not entities or status != 400:
                raise

            self.logger.warning(
                'Telegram rejected the formatting; sending it as plain text',
            )
            await self._push(surface, text, (), attempts)

        self.sent = (surface, text, entities)
        self.timer = self.clock.timer()
        self.gap = self.interval

    async def _write_loop(self) -> None:
        while True:
            await self._idle()
            wait = (self.gap - self.timer.elapsed()).total_seconds()
            if wait > 0:
                await self.sleep(wait)

            async with self.lock:
                self.dirty.clear()
                try:
                    await self._flush(final=False)
                except _Throttled as throttled:
                    self.logger.info('Holding the live reply: %s', throttled)
                    self.timer = self.clock.timer()
                    self.gap = throttled.delay
                    self.dirty.set()


class TelegramGateway(ChannelGatewayContract, ChannelMenuContract):
    logger: Logger
    client: TelegramClient
    clock: ClockContract

    def __init__(
        self,
        client: TelegramClient,
        clock: ClockContract,
        *,
        logger: Logger,
    ) -> None:
        self.logger = logger.bind(component='telegram')
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
        return await self.client.fetch_file(
            attachment.reference,
            ATTACHMENT_LIMITS[attachment.kind],
        )

    @override
    def reply(self, message: InboundMessage) -> ChannelReplyContract:
        return TelegramReply(
            self.client,
            message.chat,
            self.clock,
            private=message.private,
            logger=self.logger,
        )

    @override
    async def set_commands(
        self,
        chat: str,
        member: str | None,
        commands: tuple[tuple[str, str], ...],
        language: str = '',
    ) -> None:
        await self.client.set_my_commands(
            commands,
            scope=_command_scope(chat, member),
            language=language,
        )

    @override
    async def delete_commands(
        self,
        chat: str,
        member: str | None,
        language: str = '',
    ) -> None:
        await self.client.delete_my_commands(
            scope=_command_scope(chat, member),
            language=language,
        )


class TelegramGatewayFactory(ChannelGatewayFactoryContract):
    logger: Logger
    clock: ClockContract
    http: HTTPClientContract

    def __init__(
        self,
        http: HTTPClientContract,
        clock: ClockContract,
        *,
        logger: Logger,
    ) -> None:
        self.logger = logger.bind(component='telegram')
        self.clock = clock
        self.http = http

    @override
    def gateway(self, token: SecretStr) -> ChannelGatewayContract:
        return TelegramGateway(
            TelegramClient(http=self.http, bot_token=token),
            self.clock,
            logger=self.logger,
        )


def _command_scope(chat: str, member: str | None) -> dict[str, JsonValue]:
    if member is None:
        return {'type': 'chat', 'chat_id': chat}
    return {'type': 'chat_member', 'chat_id': chat, 'user_id': int(member)}
