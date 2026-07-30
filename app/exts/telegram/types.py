from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class TelegramValue(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True)


@dataclass(frozen=True)
class WebhookSpec:
    ack_status: int
    reject_status: int
    signature_header: str


@dataclass(frozen=True)
class MessageSpec:
    edit_interval: timedelta
    length_limit: int


@dataclass(frozen=True)
class MediaSpec:
    size_limit: int
    photo_mime: str
    audio_mime: str


WEBHOOK = WebhookSpec(
    ack_status=200,
    reject_status=403,
    signature_header='X-Telegram-Bot-Api-Secret-Token',
)
MESSAGE = MessageSpec(length_limit=4096, edit_interval=timedelta(seconds=1.1))
MEDIA = MediaSpec(
    size_limit=5 * 1024 * 1024,
    photo_mime='image/jpeg',
    audio_mime='audio/ogg',
)


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
    audio: Annotated[Voice | None, Field()] = None


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


def select_photo(
    photos: Sequence[PhotoSize],
    limit: int = MEDIA.size_limit,
) -> PhotoSize | None:
    fitting = [
        photo for photo in photos if photo.file_size is None or photo.file_size <= limit
    ]
    return max(fitting, key=lambda photo: photo.width * photo.height, default=None)
