from typing import Annotated

from app.contracts.auth import ApiClient
from app.core.dependencies import Authenticated, Provide
from lib.router import Depends, EventSourceResponse, Router

from .entities import ChatRequest, CompatibleMessage, CompatibleRequest, Completion
from .handlers import ChatHandlers

router = Router()


@router.post('/v1/chat')
async def chat(
    ask: ChatRequest,
    client: Annotated[ApiClient, Depends(Authenticated())],
    handlers: Annotated[ChatHandlers, Depends(Provide(ChatHandlers))],
) -> EventSourceResponse:
    return await handlers.native(ask, client)


# NOTE: response_model=None because the shape depends on `stream` -- fastapi cannot
# build one schema for "an SSE stream or a whole completion"
@router.post('/openai/v1/chat/completions', response_model=None)
async def completions(
    ask: CompatibleRequest,
    client: Annotated[ApiClient, Depends(Authenticated())],
    handlers: Annotated[ChatHandlers, Depends(Provide(ChatHandlers))],
) -> EventSourceResponse | Completion:
    return await handlers.openai(ask, client)


@router.post('/anthropic/v1/messages', response_model=None)
async def messages(
    ask: CompatibleRequest,
    client: Annotated[ApiClient, Depends(Authenticated())],
    handlers: Annotated[ChatHandlers, Depends(Provide(ChatHandlers))],
) -> EventSourceResponse | CompatibleMessage:
    return await handlers.anthropic(ask, client)
