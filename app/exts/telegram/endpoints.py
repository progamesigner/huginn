import logging
from collections.abc import Iterator
from typing import Annotated

from pydantic import ValidationError

from app.core.dependencies import Provide
from app.ports.channels import (
    ChannelMessageHandlerContract,
    ChannelResolverContract,
    InboundAttachment,
    InboundCommand,
    InboundMessage,
)
from lib.router import Depends, Header, Request, Response, Router

from .types import MEDIA, WEBHOOK, Message, Update, select_photo

logger = logging.getLogger(__name__)

_KIND = 'telegram'
_COMMAND_ENTITY = 'bot_command'
_COMMAND_MENTION = '@'
_COMMAND_PREFIX = '/'

router = Router()


def _attachments(message: Message) -> Iterator[InboundAttachment]:
    if (photo := select_photo(message.photo)) is not None:
        yield InboundAttachment(
            kind='image',
            mime=MEDIA.photo_mime,
            reference=photo.file_id,
            unique=photo.file_unique_id,
            height=photo.height,
            size=photo.file_size,
            width=photo.width,
        )
    elif message.photo:
        logger.warning(
            'Every size of photo message %s is over the limit',
            message.message_id,
        )

    for sound in (message.voice, message.audio):
        if sound is not None:
            yield InboundAttachment(
                kind='audio',
                mime=sound.mime_type or MEDIA.audio_mime,
                reference=sound.file_id,
                unique=sound.file_unique_id,
                duration=sound.duration,
                size=sound.file_size,
            )


def _command(message: Message) -> InboundCommand | None:
    text = message.text or ''
    entity = next(
        (
            entity
            for entity in message.entities
            if entity.type == _COMMAND_ENTITY and entity.offset == 0
        ),
        None,
    )
    if entity is None:
        return None

    name = text[: entity.length].removeprefix(_COMMAND_PREFIX)

    return InboundCommand(
        name=name.partition(_COMMAND_MENTION)[0].lower(),
        args=text[entity.length :].strip(),
    )


def _inbound(channel: str, body: bytes) -> InboundMessage | None:
    try:
        update = Update.model_validate_json(body)
    except ValidationError as error:
        logger.error('Discarding unparseable telegram update: %s', error)
        return None

    message = update.message
    if message is None:
        # NOTE: an update with no message carries nothing the app can act on
        logger.info('Discarding telegram update %s without a message', update.update_id)
        return None

    return InboundMessage(
        channel=channel,
        chat=str(message.chat.id),
        event=str(update.update_id),
        id=str(message.message_id),
        kind=_KIND,
        text=message.text or message.caption or '',
        attachments=tuple(_attachments(message)),
        command=_command(message),
        user=None if message.from_ is None else str(message.from_.id),
    )


@router.post('/webhooks/telegram')
async def telegram_webhook(
    request: Request,
    resolver: Annotated[
        ChannelResolverContract,
        Depends(Provide(ChannelResolverContract)),
    ],
    handler: Annotated[
        ChannelMessageHandlerContract,
        Depends(Provide(ChannelMessageHandlerContract)),
    ],
    secret: Annotated[
        str | None,
        Header(alias=WEBHOOK.signature_header),
    ] = None,
) -> Response:
    body = await request.body()
    channel = None if secret is None else resolver.resolve(_KIND, secret)
    if channel is None:
        return Response(status_code=WEBHOOK.reject_status)

    # NOTE: land the raw payload before parsing (webhook golden rule)
    logger.info(
        'Received telegram webhook for %s: %s',
        channel,
        body.decode('utf-8', 'replace'),
    )

    if (message := _inbound(channel, body)) is not None:
        await handler.handle(message)

    return Response(status_code=WEBHOOK.ack_status)
