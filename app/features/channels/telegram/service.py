import logging
from typing import override

from app.core.config.registry import HuginnConfig
from app.exts.telegram.gateways import build_gateway
from app.features.channels.config import CHANNELS
from app.ports.channels import (
    ChannelGatewayContract,
    ChannelMessageHandlerContract,
    InboundMessage,
)
from lib.concurrency import TaskSupervisorContract
from lib.http import HTTPClientContract
from lib.redis import RedisContract
from lib.temporal import ClockContract

from .actor import TelegramChatActor
from .config import TelegramChannelConfig

logger = logging.getLogger(__name__)

_DEDUPE_TTL = 6 * 60 * 60


class TelegramMessageHandler(ChannelMessageHandlerContract):
    actor: TelegramChatActor
    clock: ClockContract
    config: HuginnConfig
    http: HTTPClientContract
    redis: RedisContract
    supervisor: TaskSupervisorContract

    def __init__(
        self,
        config: HuginnConfig,
        redis: RedisContract,
        http: HTTPClientContract,
        clock: ClockContract,
        actor: TelegramChatActor,
        supervisor: TaskSupervisorContract,
    ) -> None:
        self.actor = actor
        self.clock = clock
        self.config = config
        self.http = http
        self.redis = redis
        self.supervisor = supervisor

    def gateway(self, channel: TelegramChannelConfig) -> ChannelGatewayContract:
        return build_gateway(self.http, channel.token, self.clock)

    @override
    async def handle(self, message: InboundMessage) -> None:
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

        self.supervisor.spawn(self._process(message, channel))

    def _channel(self, name: str) -> TelegramChannelConfig | None:
        channel = self.config.section(CHANNELS).root.get(name)
        return channel if isinstance(channel, TelegramChannelConfig) else None

    async def _process(
        self,
        message: InboundMessage,
        channel: TelegramChannelConfig,
    ) -> None:
        await self.gateway(channel).acknowledge(message)
        await self.actor.handle(channel, message)

    async def _reserve(self, message: InboundMessage) -> bool:
        key = f'telegram:{message.channel}:{message.event}'
        try:
            async with self.redis.connect() as client:
                return bool(await client.set(key, b'1', nx=True, ex=_DEDUPE_TTL))
        except Exception as error:
            # NOTE: dedupe is best-effort; availability beats exactly-once here
            logger.warning('Dedupe unavailable; processing without it: %s', error)
            return True
