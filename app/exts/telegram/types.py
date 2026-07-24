from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class TelegramValue(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True)


class User(TelegramValue):
    id: Annotated[int, Field()]
    is_bot: Annotated[bool, Field()]
    first_name: Annotated[str, Field()]
    username: Annotated[str | None, Field()] = None


class Chat(TelegramValue):
    id: Annotated[int, Field()]
    type: Annotated[str, Field()]


class PhotoSize(TelegramValue):
    file_id: Annotated[str, Field()]
    file_unique_id: Annotated[str, Field()]
    width: Annotated[int, Field()]
    height: Annotated[int, Field()]
    file_size: Annotated[int | None, Field()] = None


class Voice(TelegramValue):
    file_id: Annotated[str, Field()]
    file_unique_id: Annotated[str, Field()]
    duration: Annotated[int, Field()]
    mime_type: Annotated[str | None, Field()] = None
    file_size: Annotated[int | None, Field()] = None


class Message(TelegramValue):
    message_id: Annotated[int, Field()]
    date: Annotated[int, Field()]
    chat: Annotated[Chat, Field()]
    from_: Annotated[User | None, Field(alias='from')] = None
    text: Annotated[str | None, Field()] = None
    caption: Annotated[str | None, Field()] = None
    photo: Annotated[tuple[PhotoSize, ...], Field()] = ()
    voice: Annotated[Voice | None, Field()] = None


class Update(TelegramValue):
    update_id: Annotated[int, Field()]
    message: Annotated[Message | None, Field()] = None
    edited_message: Annotated[Message | None, Field()] = None


class File(TelegramValue):
    file_id: Annotated[str, Field()]
    file_unique_id: Annotated[str, Field()]
    file_size: Annotated[int | None, Field()] = None
    file_path: Annotated[str | None, Field()] = None


class ReactionTypeEmoji(TelegramValue):
    type: Annotated[Literal['emoji'], Field()] = 'emoji'
    emoji: Annotated[str, Field()]


class TelegramResponse(TelegramValue):
    ok: Annotated[bool, Field()]
    result: Annotated[JsonValue, Field()]
