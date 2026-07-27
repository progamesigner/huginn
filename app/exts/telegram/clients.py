import asyncio

from pydantic import JsonValue, SecretStr, TypeAdapter

from lib.http import HTTPClientContract

from .types import File, Message, ReactionTypeEmoji, TelegramResponse

_MAX_ATTEMPTS = 3


class TelegramClient:
    http: HTTPClientContract
    base_url: str

    def __init__(self, http: HTTPClientContract, bot_token: SecretStr) -> None:
        self.base_url = f'https://api.telegram.org/bot{bot_token.get_secret_value()}'
        self.http = http

    async def send_message(self, chat_id: int | str, text: str) -> Message:
        return await self._post(
            'sendMessage',
            {'chat_id': chat_id, 'text': text},
            Message,
        )

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

    async def get_file(self, file_id: str) -> File:
        return await self._post('getFile', {'file_id': file_id}, File)

    def _adapt[T](self, result: type[T], response: bytes) -> T:
        parsed = TelegramResponse.model_validate_json(response)
        return TypeAdapter(result).validate_python(parsed.result)

    async def _post[T](
        self,
        method: str,
        payload: dict[str, JsonValue],
        result: type[T],
    ) -> T:
        async with self.http.connect(self.base_url) as client:
            for attempt in range(_MAX_ATTEMPTS):
                response = await client.post(method, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt + 1 < _MAX_ATTEMPTS:
                        retry_after = 2**attempt
                        if response.status_code == 429:
                            parameters = response.json().get('parameters', {})
                            retry_after = parameters.get('retry_after', retry_after)
                        await asyncio.sleep(float(retry_after))
                        continue
                return self._adapt(result, response.raise_for_status().content)
        raise RuntimeError('telegram request exhausted retries')
