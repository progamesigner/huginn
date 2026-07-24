from typing import Annotated

from app.core.dependencies import Provide
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
        Header(alias='X-Telegram-Bot-Api-Secret-Token'),
    ] = None,
) -> Response:
    body = await request.body()
    try:
        await service.ingest(secret, body)
    except InvalidWebhookSecretError:
        return Response(status_code=403)

    return Response(status_code=200)
