from collections.abc import Sequence

from pydantic import SecretStr

from app.exts.telegram.clients import TelegramClient, webhook_url
from lib.http import HTTPClientContract
from lib.temporal import from_epoch

from .types import WebhookStatus


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


class TelegramWebhookRepository:
    http: HTTPClientContract

    def __init__(self, http: HTTPClientContract) -> None:
        self.http = http

    async def inspect(self, token: SecretStr) -> WebhookStatus:
        client = TelegramClient(http=self.http, bot_token=token)
        info = await client.get_webhook_info()

        return WebhookStatus(
            url=info.url,
            pending=info.pending_update_count,
            certificate=info.has_custom_certificate,
            allowed=info.allowed_updates,
            connections=info.max_connections,
            failed_at=(
                None
                if info.last_error_date is None
                else from_epoch(info.last_error_date)
            ),
            failure=info.last_error_message,
        )

    async def register(
        self,
        token: SecretStr,
        base_url: str,
        secret_token: SecretStr,
        allowed_updates: Sequence[str],
        drop_pending_updates: bool = False,
    ) -> str:
        client = TelegramClient(http=self.http, bot_token=token)
        url = webhook_url(base_url)
        await client.set_webhook(
            url,
            secret_token,
            allowed_updates,
            drop_pending_updates=drop_pending_updates,
        )

        return url

    async def unregister(
        self,
        token: SecretStr,
        drop_pending_updates: bool = False,
    ) -> bool:
        client = TelegramClient(http=self.http, bot_token=token)

        return await client.delete_webhook(drop_pending_updates=drop_pending_updates)
