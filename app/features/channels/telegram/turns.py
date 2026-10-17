from collections.abc import Callable

from app.contracts.channels import (
    ATTACHMENT_LIMITS,
    ChannelGatewayContract,
    ChannelReplyContract,
    InboundAttachment,
    InboundCommand,
    InboundMessage,
)
from app.features.agents.contracts import (
    ScopedCommandsServiceContract,
    TurnServiceContract,
)
from app.features.agents.entities import PromptAttachment, TurnPrompt, TurnRequest
from app.features.channels.config import routing_policy
from app.features.channels.contracts import TranscriptionServiceContract
from app.features.channels.documents import attachment_hint, read_document
from app.features.channels.exceptions import TranscriptionUnavailableError
from app.features.channels.registry import CommandRegistry
from app.features.identities.contracts import IdentityResolutionServiceContract
from app.features.identities.entities import LinkedPrincipal, Principal
from lib.documents import SUPPORTED_MIMES
from lib.observability import Logger

from .config import TelegramChannelConfig
from .menus import TelegramMenuService

type GatewayFactory = Callable[[TelegramChannelConfig], ChannelGatewayContract]

_UNATTRIBUTED = ''
_UNHEARD = 'I could not read that voice message; please send it as text.'
_UNREADABLE = 'I can only read text, photos, voice messages and supported documents.'
_TOO_LARGE = 'That file is too large for me to read.'
_UNSUPPORTED = 'I cannot read that file type.'


def _decline_note(messages: tuple[InboundMessage, ...]) -> str:
    attachments = tuple(
        attachment for message in messages for attachment in message.attachments
    )
    if any(
        attachment.size is not None
        and attachment.size > ATTACHMENT_LIMITS[attachment.kind]
        for attachment in attachments
    ):
        return _TOO_LARGE
    if any(
        attachment.kind == 'document' and attachment.mime not in SUPPORTED_MIMES
        for attachment in attachments
    ):
        return _UNSUPPORTED
    audible = any(attachment.kind == 'audio' for attachment in attachments)

    return _UNHEARD if audible else _UNREADABLE


def _prompt(
    text: str,
    attachments: tuple[PromptAttachment, ...],
    principal: Principal | None,
) -> TurnPrompt:
    if isinstance(principal, LinkedPrincipal):
        return TurnPrompt(
            text=text,
            actor=principal.stable_key,
            attachments=attachments,
            account=str(principal.user_id),
            handle=principal.handle,
            memory_scope=principal.memory_scope,
        )

    actor = _UNATTRIBUTED if principal is None else principal.stable_key
    return TurnPrompt(text=text, actor=actor, attachments=attachments)


class TelegramTurnProcessor:
    logger: Logger
    commands: CommandRegistry
    gateway: GatewayFactory
    identities: IdentityResolutionServiceContract
    transcription: TranscriptionServiceContract
    turns: TurnServiceContract

    def __init__(
        self,
        turns: TurnServiceContract,
        identities: IdentityResolutionServiceContract,
        gateway: GatewayFactory,
        transcription: TranscriptionServiceContract,
        commands: CommandRegistry,
        *,
        logger: Logger,
        scoped_commands: ScopedCommandsServiceContract | None = None,
        menus: TelegramMenuService | None = None,
    ) -> None:
        self.scoped_commands = scoped_commands
        self.menus = menus
        self.logger = logger.bind(component='telegram')
        self.commands = commands
        self.gateway = gateway
        self.identities = identities
        self.transcription = transcription
        self.turns = turns

    async def process(
        self,
        channel: TelegramChannelConfig,
        messages: tuple[InboundMessage, ...],
    ) -> None:
        if not messages:
            return

        gateway = self.gateway(channel)
        # One identity lookup per sender serves command checks and the turn.
        actors = dict[str, Principal | None]()
        scoped = set[int]()
        for index, message in enumerate(messages):
            if (command := message.command) is not None:
                if await self._scoped(channel, command, message, actors):
                    scoped.add(index)
                else:
                    await self._command(gateway, command, message, channel)
            if self.menus is not None:
                self.menus.schedule(channel, message)

        said = tuple(
            message
            for index, message in enumerate(messages)
            if message.command is None or index in scoped
        )
        if not said:
            return

        reply = gateway.reply(said[-1])
        try:
            request = await self._request(channel, gateway, said, actors)
            if not request.prompts:
                await self._decline(reply, said)
                return

            await self.turns.handle(request, reply)
        except Exception:
            self.logger.exception('Turn failed for chat %s', said[-1].chat)
            await self._abandon(gateway, reply, said)

    async def _scoped(
        self,
        channel: TelegramChannelConfig,
        command: InboundCommand,
        message: InboundMessage,
        actors: dict[str, Principal | None],
    ) -> bool:
        # Only a personal command the sender can use enters the turn pipeline;
        # anything else keeps the built-in unknown-command answer.
        if (
            self.scoped_commands is None
            or command.name in self.commands.names()
            or not self.scoped_commands.enabled(routing_policy(channel))
        ):
            return False
        try:
            key = message.user or _UNATTRIBUTED
            if key not in actors:
                actors[key] = await self._actor(channel, message)
            request = TurnRequest(
                channel=message.channel,
                chat=message.chat,
                policy=routing_policy(channel),
                prompts=(_prompt('', (), actors[key]),),
            )
            listing = await self.scoped_commands.inventory(request)
        except Exception:
            # Let the turn pipeline report the failure on this message.
            self.logger.warning(
                'Could not list personal commands for /%s',
                command.name,
            )
            return True
        return any(item.name == command.name for item in listing)

    async def _abandon(
        self,
        gateway: ChannelGatewayContract,
        reply: ChannelReplyContract,
        messages: tuple[InboundMessage, ...],
    ) -> None:
        try:
            await reply.abandon()
            for message in messages:
                await gateway.abandon(message)
        except Exception as error:
            self.logger.error('Could not report the failed turn: %s', error)

    async def _actor(
        self,
        channel: TelegramChannelConfig,
        message: InboundMessage,
    ) -> Principal | None:
        if message.user is None:
            return None

        return await self.identities.resolve(
            message.kind,
            message.channel,
            message.user,
        )

    async def _attachments(
        self,
        gateway: ChannelGatewayContract,
        message: InboundMessage,
    ) -> tuple[PromptAttachment, ...]:
        carried = []
        for attachment in message.attachments:
            if attachment.kind == 'document' and attachment.mime not in SUPPORTED_MIMES:
                continue
            if (
                attachment.size is not None
                and attachment.size > ATTACHMENT_LIMITS[attachment.kind]
            ):
                continue
            try:
                data = await gateway.fetch(attachment)
            except Exception:
                self.logger.exception(
                    'Could not fetch the %s of message %s',
                    attachment.kind,
                    message.id,
                )
                continue

            if (carrying := await self._carry(attachment, data, message)) is not None:
                carried.append(carrying)

        return tuple(carried)

    async def _carry(
        self,
        attachment: InboundAttachment,
        data: bytes,
        message: InboundMessage,
    ) -> PromptAttachment | None:
        transcript = None
        document_text = None
        if attachment.kind == 'audio':
            transcript = await self._transcript(attachment, data, message)
            if transcript is None:
                return None
        elif attachment.kind == 'document':
            reading = await read_document(
                attachment.mime,
                data,
                attachment.name,
                attachment_hint(attachment),
                logger=self.logger,
            )
            transcript = reading.transcript
            document_text = reading.text

        return PromptAttachment(
            kind=attachment.kind,
            mime=attachment.mime,
            reference=attachment.reference,
            unique=attachment.unique,
            data=data,
            hint=attachment_hint(attachment),
            name=attachment.name,
            transcript=transcript,
            document_text=document_text,
        )

    async def _command(
        self,
        gateway: ChannelGatewayContract,
        command: InboundCommand,
        message: InboundMessage,
        channel: TelegramChannelConfig | None = None,
    ) -> None:
        self.logger.info('Handling /%s for chat %s', command.name, message.chat)
        try:
            if (
                command.name == 'help'
                and self.scoped_commands is not None
                and channel is not None
            ):
                principal = await self._actor(channel, message)
                request = TurnRequest(
                    channel=message.channel,
                    chat=message.chat,
                    policy=routing_policy(channel),
                    prompts=(_prompt('', (), principal),),
                )
                listing = await self.scoped_commands.inventory(request)
                answer = '\n'.join(
                    (
                        'Here is what I understand:',
                        *(
                            f'/{item.name} — {item.description}'
                            for item in (*self.commands.listing(), *listing)
                        ),
                        'Some commands may require /help instead of appearing in the Telegram menu.',
                    ),
                )
            else:
                answer = await self.commands.dispatch(command, message)
        except Exception:
            self.logger.exception(
                'Command /%s failed for chat %s',
                command.name,
                message.chat,
            )
            await self._abandon(gateway, gateway.reply(message), (message,))
            return

        reply = gateway.reply(message)
        await reply.on_text(answer)
        await reply.finish()

    async def _decline(
        self,
        reply: ChannelReplyContract,
        messages: tuple[InboundMessage, ...],
    ) -> None:
        self.logger.info('Nothing readable in the round for chat %s', messages[-1].chat)
        await reply.on_text(_decline_note(messages))
        await reply.finish()

    async def _transcript(
        self,
        attachment: InboundAttachment,
        data: bytes,
        message: InboundMessage,
    ) -> str | None:
        try:
            return await self.transcription.transcribe(attachment.mime, data)
        except TranscriptionUnavailableError:
            self.logger.warning(
                'Dropping the voice of message %s: no transcription is configured',
                message.id,
            )
        except Exception:
            self.logger.exception(
                'Could not transcribe the voice of message %s',
                message.id,
            )

        return None

    async def _request(
        self,
        channel: TelegramChannelConfig,
        gateway: ChannelGatewayContract,
        messages: tuple[InboundMessage, ...],
        actors: dict[str, Principal | None] | None = None,
    ) -> TurnRequest:
        actors = {} if actors is None else actors
        prompts = []
        for message in messages:
            attachments = await self._attachments(gateway, message)
            if not message.text and not attachments:
                continue

            key = message.user or _UNATTRIBUTED
            if key not in actors:
                actors[key] = await self._actor(channel, message)
            text = message.text
            if message.command is not None:
                text = (
                    '/'
                    + message.command.name
                    + (' ' + message.command.args if message.command.args else '')
                )
            prompts.append(
                _prompt(text, attachments, actors[key]).model_copy(
                    update={'message_id': message.id},
                ),
            )

        return TurnRequest(
            channel=messages[-1].channel,
            chat=messages[-1].chat,
            policy=routing_policy(channel),
            prompts=tuple(prompts),
        )
