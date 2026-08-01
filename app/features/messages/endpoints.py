from typing import Annotated

from app.contracts.auth import ApiClient
from app.core.dependencies import Authenticated, Provide
from lib.router import Depends, EventSourceResponse, Router

from .handlers import ChatHandlers
from .types import ChatRequest

router = Router()


@router.post('/v1/chat')
async def chat(
    ask: ChatRequest,
    client: Annotated[ApiClient, Depends(Authenticated())],
    handlers: Annotated[ChatHandlers, Depends(Provide(ChatHandlers))],
) -> EventSourceResponse:
    return await handlers.native(ask, client)
