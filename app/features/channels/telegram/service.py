import logging
from typing import override

from pydantic import ValidationError

from app.core.channels.contracts import (
    ChannelGatewayContract,
    ChannelMessageHandlerContract,
)
from app.core.channels.types import InboundMessage
from app.core.config.registry import HuginnConfig
from app.exts.telegram.client import TelegramClient
from app.exts.telegram.gateway import TelegramGateway
from app.exts.telegram.types import Update
from app.features.channels.config import CHANNELS
from lib.http import HTTPClientContract
from lib.redis import RedisContract
from lib.temporal import ClockContract

from .actor import TelegramChatActor
from .config import TelegramChannelConfig
from .tasks import BackgroundTasks

logger = logging.getLogger(__name__)

_DEDUPE_TTL = 6 * 60 * 60


class TelegramMessageHandler(ChannelMessageHandlerContract):
    actor: TelegramChatActor
    background: BackgroundTasks
    clock: ClockContract
    config: HuginnConfig
    http: HTTPClientContract
    redis: RedisContract

    def __init__(
        self,
        config: HuginnConfig,
        redis: RedisContract,
        http: HTTPClientContract,
        clock: ClockContract,
        actor: TelegramChatActor,
        background: BackgroundTasks,
    ) -> None:
        self.actor = actor
        self.background = background
        self.clock = clock
        self.config = config
        self.http = http
        self.redis = redis

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

        self.background.spawn(self._process(message, channel))

    async def _process(
        self,
        message: InboundMessage,
        channel: TelegramChannelConfig,
    ) -> None:
        try:
            update = Update.model_validate_json(message.raw)
        except ValidationError as error:
            logger.error('Discarding unparseable telegram update: %s', error)
            return

        await self.gateway(channel).acknowledge(message)
        await self.actor.handle(message.channel, channel, update, message.raw)

    async def _reserve(self, message: InboundMessage) -> bool:
        key = f'telegram:{message.channel}:{message.event}'
        try:
            async with self.redis.connect() as client:
                return bool(await client.set(key, b'1', nx=True, ex=_DEDUPE_TTL))
        except Exception as error:
            # NOTE: dedupe is best-effort; availability beats exactly-once here
            logger.warning('Dedupe unavailable; processing without it: %s', error)
            return True

    def gateway(self, channel: TelegramChannelConfig) -> ChannelGatewayContract:
        client = TelegramClient(http=self.http, bot_token=channel.token)
        return TelegramGateway(client, self.clock)

    def _channel(self, name: str) -> TelegramChannelConfig | None:
        channel = self.config.section(CHANNELS).root.get(name)
        return channel if isinstance(channel, TelegramChannelConfig) else None
