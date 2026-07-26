import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta

from pydantic import ValidationError

from app.exts.telegram.types import Update
from lib.redis import RedisContract, RedisLock, RedisMailbox

from .config import TelegramChannelConfig

logger = logging.getLogger(__name__)

_LOCK_TTL = timedelta(seconds=30)

TurnProcessor = Callable[
    [str, TelegramChannelConfig, tuple[Update, ...]],
    Awaitable[None],
]


class TelegramChatActor:
    redis: RedisContract
    process: TurnProcessor
    ttl: timedelta

    def __init__(
        self,
        redis: RedisContract,
        process: TurnProcessor,
        *,
        ttl: timedelta = _LOCK_TTL,
    ) -> None:
        self.redis = redis
        self.process = process
        self.ttl = ttl

    async def handle(
        self,
        name: str,
        channel: TelegramChannelConfig,
        update: Update,
        body: bytes,
    ) -> None:
        message = update.message
        if message is None:
            return

        chat_id = message.chat.id
        lock = RedisLock(self.redis, self._key(name, chat_id, 'lock'), ttl=self.ttl)
        mailbox = RedisMailbox(lock, self._key(name, chat_id, 'mailbox'))

        if not await mailbox.claim(body):
            return

        async with lock.renew():
            batch: tuple[bytes, ...] = (body,)
            while batch:
                await self.process(name, channel, _parse(name, batch))
                batch = await mailbox.drain()

    def _key(self, name: str, chat_id: int, kind: str) -> str:
        return f'telegram:{name}:{kind}:{chat_id}'


def _parse(name: str, batch: tuple[bytes, ...]) -> tuple[Update, ...]:
    updates = []
    for raw in batch:
        try:
            updates.append(Update.model_validate_json(raw))
        except ValidationError:
            logger.warning('Discarding unparseable queued telegram update for %s', name)
    return tuple(updates)


async def log_turn_batch(
    name: str,
    channel: TelegramChannelConfig,
    updates: tuple[Update, ...],
) -> None:
    # NOTE: placeholder — replaces this with the real agent turn
    logger.info('Processing %d coalesced update(s) for %s', len(updates), name)
