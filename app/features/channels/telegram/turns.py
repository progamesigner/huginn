import logging
from collections.abc import Callable

from app.features.agents.contracts import TurnHandlerContract
from app.features.agents.types import PromptAttachment, TurnPrompt, TurnRequest
from app.features.channels.config import routing_policy
from app.features.identities.contracts import IdentityResolverContract
from app.ports.channels import (
    ChannelGatewayContract,
    ChannelReplyContract,
    InboundAttachment,
    InboundMessage,
)

from .config import TelegramChannelConfig

logger = logging.getLogger(__name__)

type GatewayFactory = Callable[[TelegramChannelConfig], ChannelGatewayContract]

_UNATTRIBUTED = ''
_UNREADABLE = 'I can only read text, photos and voice messages.'


def _hint(attachment: InboundAttachment) -> str:
    if attachment.width is not None and attachment.height is not None:
        detail = f' {attachment.width}x{attachment.height}'
    elif attachment.duration is not None:
        detail = f' {attachment.duration}s'
    else:
        detail = ''

    return f'[{attachment.kind}: {attachment.mime}{detail}]'


class TelegramTurnProcessor:
    gateway: GatewayFactory
    identities: IdentityResolverContract
    turns: TurnHandlerContract

    def __init__(
        self,
        turns: TurnHandlerContract,
        identities: IdentityResolverContract,
        gateway: GatewayFactory,
    ) -> None:
        self.gateway = gateway
        self.identities = identities
        self.turns = turns

    async def process(
        self,
        channel: TelegramChannelConfig,
        messages: tuple[InboundMessage, ...],
    ) -> None:
        if not messages:
            return

        gateway = self.gateway(channel)
        reply = gateway.reply(messages[-1])
        try:
            request = await self._request(channel, gateway, messages)
            if not request.prompts:
                await self._decline(reply, messages)
                return

            await self.turns.handle(request, reply)
        except Exception:
            logger.exception('Turn failed for chat %s', messages[-1].chat)
            await self._abandon(gateway, reply, messages)

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
            channel.kind,
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

            carried.append(
                PromptAttachment(
                    kind=attachment.kind,
                    mime=attachment.mime,
                    reference=attachment.reference,
                    unique=attachment.unique,
                    data=data,
                    hint=_hint(attachment),
                ),
            )

        return tuple(carried)

    async def _decline(
        self,
        reply: ChannelReplyContract,
        messages: tuple[InboundMessage, ...],
    ) -> None:
        logger.info('Nothing readable in the round for chat %s', messages[-1].chat)
        await reply.on_text(_UNREADABLE)
        await reply.finish()

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
