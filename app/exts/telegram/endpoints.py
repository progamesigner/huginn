import logging
from typing import Annotated

from pydantic import ValidationError

from app.core.dependencies import Provide
from app.ports.channels import (
    ChannelMessageHandlerContract,
    ChannelResolverContract,
    InboundMessage,
)
from lib.router import Depends, Header, Request, Response, Router

from .types import WEBHOOK, Update

logger = logging.getLogger(__name__)

_KIND = 'telegram'

router = Router()


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
        raw=body,
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
