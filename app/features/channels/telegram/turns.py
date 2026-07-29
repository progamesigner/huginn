import logging
from collections.abc import Callable

from app.features.agents.contracts import TurnHandlerContract
from app.features.agents.types import TurnPrompt, TurnRequest
from app.features.channels.config import routing_policy
from app.features.identities.contracts import IdentityResolverContract
from app.ports.channels import (
    ChannelGatewayContract,
    ChannelReplyContract,
    InboundMessage,
)

from .config import TelegramChannelConfig

logger = logging.getLogger(__name__)

type GatewayFactory = Callable[[TelegramChannelConfig], ChannelGatewayContract]

_UNATTRIBUTED = ''


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
            await self.turns.handle(await self._request(channel, messages), reply)
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

    async def _request(
        self,
        channel: TelegramChannelConfig,
        messages: tuple[InboundMessage, ...],
    ) -> TurnRequest:
        actors: dict[str, str] = {}
        prompts = []
        for message in messages:
            if not message.text:
                continue

            key = message.user or _UNATTRIBUTED
            if key not in actors:
                actors[key] = await self._actor(channel, message)
            prompts.append(TurnPrompt(text=message.text, actor=actors[key]))

        return TurnRequest(
            channel=messages[-1].channel,
            chat=messages[-1].chat,
            policy=routing_policy(channel),
            prompts=tuple(prompts),
        )
