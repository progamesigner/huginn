import logging
from abc import ABC, abstractmethod
from typing import override

from app.exts.telegram.types import Update

from .config import TelegramChannelConfig

logger = logging.getLogger(__name__)


class TelegramUpdateHandler(ABC):
    @abstractmethod
    async def handle(
        self,
        name: str,
        channel: TelegramChannelConfig,
        update: Update,
    ) -> None: ...


class LoggingUpdateHandler(TelegramUpdateHandler):
    @override
    async def handle(
        self,
        name: str,
        channel: TelegramChannelConfig,
        update: Update,
    ) -> None:
        logger.info(
            'Processing telegram update %s for channel %s',
            update.update_id,
            name,
        )
