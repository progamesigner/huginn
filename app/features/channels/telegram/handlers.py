from abc import ABC, abstractmethod
from typing import override

from app.exts.telegram.client import TelegramClient
from app.exts.telegram.types import Update
from lib.http import HTTPClientContract

from .config import TelegramChannelConfig

_ACK_EMOJI = '👀'


class TelegramUpdateHandler(ABC):
    @abstractmethod
    async def handle(
        self,
        name: str,
        channel: TelegramChannelConfig,
        update: Update,
    ) -> None: ...


class ReactionUpdateHandler(TelegramUpdateHandler):
    http: HTTPClientContract

    def __init__(self, http: HTTPClientContract) -> None:
        self.http = http

    @override
    async def handle(
        self,
        name: str,
        channel: TelegramChannelConfig,
        update: Update,
    ) -> None:
        message = update.message
        if message is None:
            return

        client = TelegramClient(http=self.http, bot_token=channel.token)
        await client.set_message_reaction(
            message.chat.id,
            message.message_id,
            _ACK_EMOJI,
        )
