from collections.abc import Sequence

from pydantic import SecretStr

from app.exts.telegram.clients import TelegramClient
from lib.http import HTTPClientContract


class TelegramMenuRepository:
    http: HTTPClientContract

    def __init__(self, http: HTTPClientContract) -> None:
        self.http = http

    async def publish(
        self,
        token: SecretStr,
        commands: Sequence[tuple[str, str]],
    ) -> bool:
        client = TelegramClient(http=self.http, bot_token=token)

        return await client.set_my_commands(commands)
