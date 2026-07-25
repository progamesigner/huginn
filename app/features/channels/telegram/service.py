import hmac
import logging

from pydantic import ValidationError

from app.core.config.registry import HuginnConfig
from app.exts.telegram.types import Update
from app.features.channels.config import CHANNELS
from lib.redis import RedisContract

from .actor import TelegramChatActor
from .config import TelegramChannelConfig
from .exceptions import InvalidWebhookSecretError
from .handlers import TelegramUpdateHandler
from .tasks import BackgroundTasks

logger = logging.getLogger(__name__)

_DEDUPE_TTL = 6 * 60 * 60


class TelegramWebhookService:
    actor: TelegramChatActor
    background: BackgroundTasks
    config: HuginnConfig
    handler: TelegramUpdateHandler
    redis: RedisContract

    def __init__(
        self,
        config: HuginnConfig,
        redis: RedisContract,
        handler: TelegramUpdateHandler,
        actor: TelegramChatActor,
        background: BackgroundTasks,
    ) -> None:
        self.actor = actor
        self.background = background
        self.config = config
        self.handler = handler
        self.redis = redis

    async def ingest(self, secret: str | None, body: bytes) -> None:
        name, channel = self._resolve(secret)

        # NOTE: land the raw payload before parsing (webhook golden rule)
        logger.info(
            'Received telegram webhook for %s: %s',
            name,
            body.decode('utf-8', 'replace'),
        )

        try:
            update = Update.model_validate_json(body)
        except ValidationError as error:
            logger.error('Discarding unparseable telegram update: %s', error)
            return None

        if not await self._reserve(name, update.update_id):
            logger.info('Skipping duplicate update %s for %s', update.update_id, name)
            return None

        self.background.spawn(self._process(name, channel, update, body))

    async def _process(
        self,
        name: str,
        channel: TelegramChannelConfig,
        update: Update,
        body: bytes,
    ) -> None:
        await self.handler.handle(name, channel, update)
        await self.actor.handle(name, channel, update, body)

    async def _reserve(self, name: str, update_id: int) -> bool:
        key = f'telegram:{name}:{update_id}'
        try:
            async with self.redis.connect() as client:
                return bool(await client.set(key, b'1', nx=True, ex=_DEDUPE_TTL))
        except Exception as error:
            # NOTE: dedupe is best-effort; availability beats exactly-once here
            logger.warning('Dedupe unavailable; processing without it: %s', error)
            return True

    def _resolve(self, secret: str | None) -> tuple[str, TelegramChannelConfig]:
        if secret is None:
            raise InvalidWebhookSecretError('missing secret token header')

        for name, channel in self.config.section(CHANNELS).root.items():
            if isinstance(channel, TelegramChannelConfig) and hmac.compare_digest(
                channel.secret_token.get_secret_value(),
                secret,
            ):
                return name, channel

        raise InvalidWebhookSecretError('no channel matches the provided secret token')
