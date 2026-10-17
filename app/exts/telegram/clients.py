import asyncio
from collections.abc import Mapping, Sequence

import httpx
from pydantic import JsonValue, SecretStr, TypeAdapter

from app.contracts.channels import ATTACHMENT_LIMITS
from lib.http import HTTPClientContract

from .exceptions import MediaTooLargeError, MediaUnavailableError
from .types import (
    WEBHOOK,
    BotCommand,
    File,
    Message,
    MessageEntity,
    ReactionTypeEmoji,
    TelegramResponse,
    Upload,
    WebhookInfo,
)

_MAX_ATTEMPTS = 3
_UPLOAD_METHODS: dict[Upload, str] = {
    'audio': 'sendAudio',
    'document': 'sendDocument',
    'photo': 'sendPhoto',
    'video': 'sendVideo',
    'voice': 'sendVoice',
}

type Multipart = dict[str, tuple[str, bytes, str]]


def webhook_url(base_url: str) -> str:
    return f'{base_url.rstrip("/")}{WEBHOOK.path}'


def _formatting(entities: Sequence[MessageEntity]) -> dict[str, JsonValue]:
    if not entities:
        return {}

    return {'entities': [entity.model_dump(exclude_none=True) for entity in entities]}


# NOTE: httpx's message names the full url, which carries the bot token
def _checked(response: httpx.Response, path: str) -> httpx.Response:
    try:
        return response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise httpx.HTTPStatusError(
            f'telegram answered {response.status_code} to {path}',
            request=error.request,
            response=error.response,
        ) from None


class TelegramClient:
    http: HTTPClientContract
    base_url: str
    file_url: str

    def __init__(self, http: HTTPClientContract, bot_token: SecretStr) -> None:
        token = bot_token.get_secret_value()
        self.base_url = f'https://api.telegram.org/bot{token}'
        self.file_url = f'https://api.telegram.org/file/bot{token}'
        self.http = http

    async def delete_webhook(self, drop_pending_updates: bool = False) -> bool:
        return await self._post(
            'deleteWebhook',
            {'drop_pending_updates': drop_pending_updates},
            bool,
        )

    async def download_file(self, file_path: str) -> bytes:
        return await self._request(self.file_url, file_path)

    async def edit_message_text(
        self,
        chat_id: int | str,
        message_id: int,
        text: str,
        entities: Sequence[MessageEntity] = (),
    ) -> Message:
        return await self._post(
            'editMessageText',
            {
                'chat_id': chat_id,
                'message_id': message_id,
                'text': text,
                **_formatting(entities),
            },
            Message,
        )

    async def fetch_file(
        self,
        file_id: str,
        limit: int = ATTACHMENT_LIMITS['image'],
    ) -> bytes:
        file = await self.get_file(file_id)
        if file.file_path is None:
            raise MediaUnavailableError(f'telegram file {file_id} has no download path')

        if file.file_size is not None and file.file_size > limit:
            raise MediaTooLargeError(
                f'telegram file {file_id} is {file.file_size} bytes, over {limit}',
            )

        content = await self.download_file(file.file_path)
        if len(content) > limit:
            raise MediaTooLargeError(
                f'telegram file {file_id} downloaded {len(content)} bytes,'
                f' over {limit}',
            )

        return content

    async def get_file(self, file_id: str) -> File:
        return await self._post('getFile', {'file_id': file_id}, File)

    async def get_webhook_info(self) -> WebhookInfo:
        return await self._post('getWebhookInfo', {}, WebhookInfo)

    async def send_chat_action(self, chat_id: int | str, action: str) -> bool:
        return await self._post(
            'sendChatAction',
            {'chat_id': chat_id, 'action': action},
            bool,
        )

    async def send_file(
        self,
        chat_id: int | str,
        upload: Upload,
        name: str,
        data: bytes,
        mime: str,
    ) -> Message:
        response = await self._request(
            self.base_url,
            _UPLOAD_METHODS[upload],
            {'chat_id': str(chat_id)},
            files={upload: (name, data, mime)},
        )
        parsed = TelegramResponse.model_validate_json(response)
        return Message.model_validate(parsed.result)

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        entities: Sequence[MessageEntity] = (),
    ) -> Message:
        return await self._post(
            'sendMessage',
            {'chat_id': chat_id, 'text': text, **_formatting(entities)},
            Message,
        )

    async def set_my_commands(
        self,
        commands: Sequence[tuple[str, str]],
        *,
        scope: Mapping[str, JsonValue] | None = None,
        language: str = '',
    ) -> bool:
        menu = [
            BotCommand(command=name, description=description)
            for name, description in commands
        ]
        return await self._post(
            'setMyCommands',
            {
                'commands': [entry.model_dump() for entry in menu],
                **({'scope': dict(scope)} if scope is not None else {}),
                **({'language_code': language} if language else {}),
            },
            bool,
        )

    async def delete_my_commands(
        self,
        *,
        scope: Mapping[str, JsonValue],
        language: str = '',
    ) -> bool:
        return await self._post(
            'deleteMyCommands',
            {'scope': dict(scope), **({'language_code': language} if language else {})},
            bool,
        )

    async def set_message_reaction(
        self,
        chat_id: int | str,
        message_id: int | str,
        emoji: str,
    ) -> bool:
        reaction = ReactionTypeEmoji(emoji=emoji)
        return await self._post(
            'setMessageReaction',
            {
                'chat_id': chat_id,
                'message_id': message_id,
                'reaction': [reaction.model_dump()],
            },
            bool,
        )

    async def set_webhook(
        self,
        url: str,
        secret_token: SecretStr,
        allowed_updates: Sequence[str],
        drop_pending_updates: bool = False,
    ) -> bool:
        return await self._post(
            'setWebhook',
            {
                'url': url,
                'secret_token': secret_token.get_secret_value(),
                'allowed_updates': list(allowed_updates),
                'drop_pending_updates': drop_pending_updates,
            },
            bool,
        )

    async def _post[T](
        self,
        method: str,
        payload: dict[str, JsonValue],
        result: type[T],
    ) -> T:
        response = await self._request(self.base_url, method, payload)
        parsed = TelegramResponse.model_validate_json(response)
        return TypeAdapter(result).validate_python(parsed.result)

    async def _request(
        self,
        base_url: str,
        path: str,
        payload: dict[str, JsonValue] | None = None,
        *,
        files: Multipart | None = None,
    ) -> bytes:
        async with self.http.connect(base_url) as client:
            for attempt in range(_MAX_ATTEMPTS):
                if files is not None:
                    response = await client.post(path, data=payload, files=files)
                elif payload is not None:
                    response = await client.post(path, json=payload)
                else:
                    response = await client.get(path)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt + 1 < _MAX_ATTEMPTS:
                        retry_after = 2**attempt
                        if response.status_code == 429:
                            parameters = response.json().get('parameters', {})
                            retry_after = parameters.get('retry_after', retry_after)
                        await asyncio.sleep(float(retry_after))
                        continue
                return _checked(response, path).content
        raise RuntimeError('telegram request exhausted retries')
