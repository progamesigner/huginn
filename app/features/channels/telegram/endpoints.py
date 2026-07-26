from typing import Annotated

from app.core.dependencies import Provide
from app.exts.telegram.types import WEBHOOK
from lib.router import Depends, Header, Request, Response, Router

from .exceptions import InvalidWebhookSecretError
from .service import TelegramWebhookService

router = Router()


@router.post('/webhooks/telegram')
async def telegram_webhook(
    request: Request,
    service: Annotated[
        TelegramWebhookService,
        Depends(Provide(TelegramWebhookService)),
    ],
    secret: Annotated[
        str | None,
        Header(alias=WEBHOOK.signature_header),
    ] = None,
) -> Response:
    body = await request.body()
    try:
        await service.ingest(secret, body)
    except InvalidWebhookSecretError:
        return Response(status_code=WEBHOOK.reject_status)

    return Response(status_code=WEBHOOK.ack_status)
