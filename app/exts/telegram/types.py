from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from itertools import accumulate
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from telegramify_markdown import convert, utf16_len

from app.contracts.channels import ATTACHMENT_LIMITS, ArtifactKind


class TelegramValue(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True, hide_input_in_errors=True)


@dataclass(frozen=True)
class WebhookSpec:
    ack_status: int
    path: str
    reject_status: int
    signature_header: str


@dataclass(frozen=True)
class MessageSpec:
    edit_interval: timedelta
    length_limit: int


@dataclass(frozen=True)
class MediaSpec:
    photo_mime: str
    audio_mime: str


@dataclass(frozen=True)
class UploadSpec:
    file_limit: int
    photo_limit: int
    photo_mimes: frozenset[str]


type Upload = Literal['audio', 'document', 'photo', 'video', 'voice']


@dataclass(frozen=True)
class TypingSpec:
    action: str
    refresh_interval: timedelta


WEBHOOK = WebhookSpec(
    ack_status=200,
    path='/webhooks/telegram',
    reject_status=403,
    signature_header='X-Telegram-Bot-Api-Secret-Token',
)
MESSAGE = MessageSpec(length_limit=4096, edit_interval=timedelta(seconds=1.1))
MEDIA = MediaSpec(
    photo_mime='image/jpeg',
    audio_mime='audio/ogg',
)
UPLOAD = UploadSpec(
    file_limit=50 * 1024 * 1024,
    photo_limit=10 * 1024 * 1024,
    photo_mimes=frozenset({'image/jpeg', 'image/png', 'image/webp'}),
)
TYPING = TypingSpec(action='typing', refresh_interval=timedelta(seconds=4))


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


class Audio(TelegramValue):
    file_id: Annotated[str, Field()]
    file_unique_id: Annotated[str, Field()]
    duration: Annotated[int, Field()]
    mime_type: Annotated[str | None, Field()] = None
    file_size: Annotated[int | None, Field()] = None


class Document(TelegramValue):
    file_id: Annotated[str, Field()]
    file_unique_id: Annotated[str, Field()]
    mime_type: Annotated[str | None, Field()] = None
    file_name: Annotated[str | None, Field()] = None
    file_size: Annotated[int | None, Field()] = None


class MessageEntity(TelegramValue):
    type: Annotated[str, Field()]
    offset: Annotated[int, Field()]
    length: Annotated[int, Field()]
    url: Annotated[str | None, Field()] = None
    language: Annotated[str | None, Field()] = None
    custom_emoji_id: Annotated[str | None, Field()] = None


class Message(TelegramValue):
    message_id: Annotated[int, Field()]
    date: Annotated[int, Field()]
    chat: Annotated[Chat, Field()]
    from_: Annotated[User | None, Field(alias='from')] = None
    text: Annotated[str | None, Field()] = None
    caption: Annotated[str | None, Field()] = None
    entities: Annotated[tuple[MessageEntity, ...], Field()] = ()
    photo: Annotated[tuple[PhotoSize, ...], Field()] = ()
    audio: Annotated[Audio | None, Field()] = None
    voice: Annotated[Voice | None, Field()] = None
    document: Annotated[Document | None, Field()] = None


class Update(TelegramValue):
    update_id: Annotated[int, Field()]
    message: Annotated[Message | None, Field()] = None
    edited_message: Annotated[Message | None, Field()] = None


class File(TelegramValue):
    file_id: Annotated[str, Field()]
    file_unique_id: Annotated[str, Field()]
    file_size: Annotated[int | None, Field()] = None
    file_path: Annotated[str | None, Field()] = None


class BotCommand(TelegramValue):
    command: Annotated[str, Field()]
    description: Annotated[str, Field()]


class WebhookInfo(TelegramValue):
    url: Annotated[str, Field()]
    has_custom_certificate: Annotated[bool, Field()]
    pending_update_count: Annotated[int, Field()]
    allowed_updates: Annotated[tuple[str, ...], Field()] = ()
    last_error_date: Annotated[int | None, Field()] = None
    last_error_message: Annotated[str | None, Field()] = None
    max_connections: Annotated[int | None, Field()] = None


class ReactionTypeEmoji(TelegramValue):
    type: Annotated[Literal['emoji'], Field()] = 'emoji'
    emoji: Annotated[str, Field()]


class TelegramResponse(TelegramValue):
    ok: Annotated[bool, Field()]
    result: Annotated[JsonValue, Field()]


def clip_entities(
    entities: Sequence[MessageEntity],
    start: int,
    end: int,
) -> tuple[MessageEntity, ...]:
    bounded = [
        (entity, max(entity.offset, start), min(entity.offset + entity.length, end))
        for entity in entities
    ]
    return tuple(
        entity.model_copy(
            update={'offset': begin - start, 'length': finish - begin},
        )
        for entity, begin, finish in bounded
        if finish > begin
    )


def render(markdown: str) -> tuple[str, tuple[MessageEntity, ...]]:
    text, entities = convert(markdown, latex_escape=True)
    return text, tuple(
        MessageEntity(
            type=entity.type,
            offset=entity.offset,
            length=entity.length,
            url=entity.url,
            language=entity.language,
            custom_emoji_id=entity.custom_emoji_id,
        )
        for entity in entities
    )


def select_photo(
    photos: Sequence[PhotoSize],
    limit: int = ATTACHMENT_LIMITS['image'],
) -> PhotoSize | None:
    fitting = [
        photo for photo in photos if photo.file_size is None or photo.file_size <= limit
    ]
    return max(fitting, key=lambda photo: photo.width * photo.height, default=None)


def utf16_offsets(text: str) -> tuple[int, ...]:
    return tuple(accumulate((utf16_len(character) for character in text), initial=0))


# NOTE: the mime picks the method, not the kind alone -- telegram plays only ogg/opus
# as a voice note and shows only a few image formats inline
def select_upload(kind: ArtifactKind, mime: str) -> Upload:
    match kind:
        case 'image' if mime in UPLOAD.photo_mimes:
            return 'photo'
        case 'audio' if mime == MEDIA.audio_mime:
            return 'voice'
        case 'audio':
            return 'audio'
        case 'video':
            return 'video'
        case _:
            return 'document'


def upload_limit(upload: Upload) -> int:
    return UPLOAD.photo_limit if upload == 'photo' else UPLOAD.file_limit
