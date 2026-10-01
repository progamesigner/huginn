from collections.abc import Awaitable, Callable
from datetime import timedelta

from pydantic import ValidationError

from app.contracts.channels import InboundMessage
from lib.observability import Logger
from lib.redis import RedisContract, RedisLock, RedisMailbox

from .config import TelegramChannelConfig

_LOCK_TTL = timedelta(seconds=30)

TurnProcessor = Callable[
    [TelegramChannelConfig, tuple[InboundMessage, ...]],
    Awaitable[None],
]


class TelegramChatActor:
    logger: Logger
    redis: RedisContract
    process: TurnProcessor
    ttl: timedelta

    def __init__(
        self,
        redis: RedisContract,
        process: TurnProcessor,
        *,
        ttl: timedelta = _LOCK_TTL,
        logger: Logger,
    ) -> None:
        self.logger = logger.bind(component='telegram')
        self.redis = redis
        self.process = process
        self.ttl = ttl

    async def handle(
        self,
        channel: TelegramChannelConfig,
        message: InboundMessage,
    ) -> None:
        lock = RedisLock(
            self.redis,
            self._key(message, 'lock'),
            ttl=self.ttl,
            logger=self.logger,
        )
        mailbox = RedisMailbox(lock, self._key(message, 'mailbox'))

        # NOTE: the queue carries the contract's own serialization, so a drained round is
        # read back as our vocabulary and the vendor's payload never reaches the feature
        if not await mailbox.claim(message.model_dump_json().encode()):
            return

        async with lock.renew():
            await self.process(channel, (message,))
            while batch := await mailbox.drain():
                await self.process(
                    channel,
                    _parse(message.channel, batch, logger=self.logger),
                )

    def _key(self, message: InboundMessage, kind: str) -> str:
        return f'telegram:{message.channel}:{kind}:{message.chat}'


def _parse(
    name: str,
    batch: tuple[bytes, ...],
    *,
    logger: Logger,
) -> tuple[InboundMessage, ...]:
    messages = []
    for raw in batch:
        try:
            messages.append(InboundMessage.model_validate_json(raw))
        except ValidationError:
            logger.warning('Discarding unreadable queued message for %s', name)
    return tuple(messages)
