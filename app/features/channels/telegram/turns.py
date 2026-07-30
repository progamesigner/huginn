import logging
from collections.abc import Callable

from app.features.agents.contracts import TurnHandlerContract
from app.features.agents.types import PromptAttachment, TurnPrompt, TurnRequest
from app.features.channels.config import routing_policy
from app.features.channels.contracts import TranscriptionContract
from app.features.channels.exceptions import TranscriptionUnavailableError
from app.features.channels.registry import CommandRegistry
from app.features.identities.contracts import IdentityResolverContract
from app.ports.channels import (
    ChannelGatewayContract,
    ChannelReplyContract,
    InboundAttachment,
    InboundCommand,
    InboundMessage,
)

from .config import TelegramChannelConfig

logger = logging.getLogger(__name__)

type GatewayFactory = Callable[[TelegramChannelConfig], ChannelGatewayContract]

_UNATTRIBUTED = ''
_UNHEARD = 'I could not read that voice message; please send it as text.'
_UNREADABLE = 'I can only read text, photos and voice messages.'


def _decline_note(messages: tuple[InboundMessage, ...]) -> str:
    audible = any(
        attachment.kind == 'audio'
        for message in messages
        for attachment in message.attachments
    )

    return _UNHEARD if audible else _UNREADABLE


def _hint(attachment: InboundAttachment) -> str:
    if attachment.width is not None and attachment.height is not None:
        detail = f' {attachment.width}x{attachment.height}'
    elif attachment.duration is not None:
        detail = f' {attachment.duration}s'
    else:
        detail = ''

    return f'[{attachment.kind}: {attachment.mime}{detail}]'


class TelegramTurnProcessor:
    commands: CommandRegistry
    gateway: GatewayFactory
    identities: IdentityResolverContract
    transcription: TranscriptionContract
    turns: TurnHandlerContract

    def __init__(
        self,
        turns: TurnHandlerContract,
        identities: IdentityResolverContract,
        gateway: GatewayFactory,
        transcription: TranscriptionContract,
        commands: CommandRegistry,
    ) -> None:
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
        for message in messages:
            if (command := message.command) is not None:
                await self._command(gateway, command, message)

        said = tuple(message for message in messages if message.command is None)
        if not said:
            return

        reply = gateway.reply(said[-1])
        try:
            request = await self._request(channel, gateway, said)
            if not request.prompts:
                await self._decline(reply, said)
                return

            await self.turns.handle(request, reply)
        except Exception:
            logger.exception('Turn failed for chat %s', said[-1].chat)
            await self._abandon(gateway, reply, said)

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
            logger.error('Could not report the failed turn: %s', error)

    async def _actor(
        self,
        channel: TelegramChannelConfig,
        message: InboundMessage,
    ) -> str:
        if message.user is None:
            return _UNATTRIBUTED

        principal = await self.identities.resolve(
            message.kind,
            message.channel,
            message.user,
        )
        return principal.stable_key

    async def _attachments(
        self,
        gateway: ChannelGatewayContract,
        message: InboundMessage,
    ) -> tuple[PromptAttachment, ...]:
        carried = []
        for attachment in message.attachments:
            try:
                data = await gateway.fetch(attachment)
            except Exception:
                logger.exception(
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
        if attachment.kind == 'audio':
            transcript = await self._transcript(attachment, data, message)
            if transcript is None:
                return None

        return PromptAttachment(
            kind=attachment.kind,
            mime=attachment.mime,
            reference=attachment.reference,
            unique=attachment.unique,
            data=data,
            hint=_hint(attachment),
            transcript=transcript,
        )

    async def _command(
        self,
        gateway: ChannelGatewayContract,
        command: InboundCommand,
        message: InboundMessage,
    ) -> None:
        logger.info('Handling /%s for chat %s', command.name, message.chat)
        try:
            answer = await self.commands.dispatch(command, message)
        except Exception:
            logger.exception(
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
        logger.info('Nothing readable in the round for chat %s', messages[-1].chat)
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
            logger.warning(
                'Dropping the voice of message %s: no transcription is configured',
                message.id,
            )
        except Exception:
            logger.exception('Could not transcribe the voice of message %s', message.id)

        return None

    async def _request(
        self,
        channel: TelegramChannelConfig,
        gateway: ChannelGatewayContract,
        messages: tuple[InboundMessage, ...],
    ) -> TurnRequest:
        actors = {}
        prompts = []
        for message in messages:
            attachments = await self._attachments(gateway, message)
            if not message.text and not attachments:
                continue

            key = message.user or _UNATTRIBUTED
            if key not in actors:
                actors[key] = await self._actor(channel, message)
            prompts.append(
                TurnPrompt(
                    text=message.text,
                    actor=actors[key],
                    attachments=attachments,
                ),
            )

        return TurnRequest(
            channel=messages[-1].channel,
            chat=messages[-1].chat,
            policy=routing_policy(channel),
            prompts=tuple(prompts),
        )
