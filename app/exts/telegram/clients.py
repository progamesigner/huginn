import asyncio

from pydantic import JsonValue, SecretStr, TypeAdapter

from lib.http import HTTPClientContract

from .exceptions import MediaTooLargeError, MediaUnavailableError
from .types import MEDIA, File, Message, ReactionTypeEmoji, TelegramResponse

_MAX_ATTEMPTS = 3


class TelegramClient:
    http: HTTPClientContract
    base_url: str
    file_url: str

    def __init__(self, http: HTTPClientContract, bot_token: SecretStr) -> None:
        token = bot_token.get_secret_value()
        self.base_url = f'https://api.telegram.org/bot{token}'
        self.file_url = f'https://api.telegram.org/file/bot{token}'
        self.http = http

    async def download_file(self, file_path: str) -> bytes:
        return await self._request(self.file_url, file_path)

    async def edit_message_text(
        self,
        chat_id: int | str,
        message_id: int,
        text: str,
    ) -> Message:
        return await self._post(
            'editMessageText',
            {'chat_id': chat_id, 'message_id': message_id, 'text': text},
            Message,
        )

    async def fetch_file(self, file_id: str, limit: int = MEDIA.size_limit) -> bytes:
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

    async def send_message(self, chat_id: int | str, text: str) -> Message:
        return await self._post(
            'sendMessage',
            {'chat_id': chat_id, 'text': text},
            Message,
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
    ) -> bytes:
        async with self.http.connect(base_url) as client:
            for attempt in range(_MAX_ATTEMPTS):
                response = (
                    await client.get(path)
                    if payload is None
                    else await client.post(path, json=payload)
                )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt + 1 < _MAX_ATTEMPTS:
                        retry_after = 2**attempt
                        if response.status_code == 429:
                            parameters = response.json().get('parameters', {})
                            retry_after = parameters.get('retry_after', retry_after)
                        await asyncio.sleep(float(retry_after))
                        continue
                return response.raise_for_status().content
        raise RuntimeError('telegram request exhausted retries')
