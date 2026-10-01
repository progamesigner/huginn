import logging
from typing import override
from uuid import uuid4

from structlog.contextvars import bound_contextvars

from app.contracts.channels import (
    ChannelGatewayContract,
    ChannelGatewayFactoryContract,
    ChannelMessageHandlerContract,
    InboundMessage,
)
from app.core.config.registry import HuginnConfig
from app.features.channels.config import CHANNELS
from lib.concurrency import TaskSupervisorContract
from lib.redis import RedisContract
from lib.temporal import ClockContract, Timer

from .actor import TelegramChatActor
from .config import TelegramChannelConfig

logger = logging.getLogger(__name__)

_DEDUPE_TTL = 6 * 60 * 60


class TelegramMessageHandler(ChannelMessageHandlerContract):
    actor: TelegramChatActor
    clock: ClockContract
    config: HuginnConfig
    gateways: ChannelGatewayFactoryContract
    redis: RedisContract
    supervisor: TaskSupervisorContract

    def __init__(
        self,
        config: HuginnConfig,
        redis: RedisContract,
        gateways: ChannelGatewayFactoryContract,
        clock: ClockContract,
        actor: TelegramChatActor,
        supervisor: TaskSupervisorContract,
    ) -> None:
        self.actor = actor
        self.clock = clock
        self.config = config
        self.gateways = gateways
        self.redis = redis
        self.supervisor = supervisor

    def gateway(self, channel: TelegramChannelConfig) -> ChannelGatewayContract:
        return self.gateways.gateway(channel.token)

    @override
    async def handle(self, message: InboundMessage) -> None:
        # NOTE: the clock starts here, before dedupe, so the measured acknowledgement
        # covers everything this service does rather than only the vendor call
        timer = self.clock.timer()
        channel = self._channel(message.channel)
        if channel is None:
            logger.error('No telegram channel named %s', message.channel)
            return

        if not await self._reserve(message):
            logger.info(
                'Skipping duplicate update %s for %s',
                message.event,
                message.channel,
            )
            return

        self.supervisor.spawn(self._process(message, channel, timer))

    async def _abandon(
        self,
        gateway: ChannelGatewayContract,
        message: InboundMessage,
    ) -> None:
        try:
            await gateway.abandon(message)
        except Exception as error:
            logger.error('Could not report the failed turn: %s', error)

    async def _acknowledge(
        self,
        gateway: ChannelGatewayContract,
        message: InboundMessage,
        timer: Timer,
    ) -> None:
        try:
            await gateway.acknowledge(message)
        except Exception as error:
            logger.warning('Could not acknowledge %s: %s', message.event, error)
        else:
            logger.info(
                'Acknowledged %s after %.3fs',
                message.event,
                timer.elapsed().total_seconds(),
            )

    def _channel(self, name: str) -> TelegramChannelConfig | None:
        channel = self.config.section(CHANNELS).root.get(name)
        return channel if isinstance(channel, TelegramChannelConfig) else None

    async def _process(
        self,
        message: InboundMessage,
        channel: TelegramChannelConfig,
        timer: Timer,
    ) -> None:
        gateway = self.gateway(channel)
        with bound_contextvars(turn_id=uuid4().hex):
            await self._acknowledge(gateway, message, timer)
            try:
                await self.actor.handle(channel, message)
            except Exception:
                logger.exception('Turn failed for %s', message.channel)
                await self._abandon(gateway, message)

    async def _reserve(self, message: InboundMessage) -> bool:
        key = f'telegram:{message.channel}:{message.event}'
        try:
            async with self.redis.connect() as client:
                return bool(await client.set(key, b'1', nx=True, ex=_DEDUPE_TTL))
        except Exception as error:
            # NOTE: dedupe is best-effort; availability beats exactly-once here
            logger.warning('Dedupe unavailable; processing without it: %s', error)
            return True
