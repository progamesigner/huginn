from collections.abc import AsyncGenerator
from contextlib import aclosing

from app.contracts.auth import ApiClient
from lib.router import EventSourceResponse

from .presenters import NativePresenter, Presenter, encode
from .services import ChatService, PreparedTurn
from .types import ChatRequest

# NOTE: a proxy that buffers a stream turns it back into one late answer, and the nginx
# ingress buffers by default
_HEADERS = {'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}


class ChatHandlers:
    service: ChatService

    def __init__(self, service: ChatService) -> None:
        self.service = service

    async def native(self, ask: ChatRequest, client: ApiClient) -> EventSourceResponse:
        # NOTE: prepare runs here rather than inside the generator, because a generator
        # body does not run until the response head is already on the wire -- and a
        # refusal after that can no longer be a status code
        prepared = await self.service.prepare(
            client,
            ask.message,
            conversation=ask.conversation,
            agent=ask.agent,
        )

        return EventSourceResponse(
            self.frames(prepared, NativePresenter()),
            headers=_HEADERS,
        )

    async def frames(
        self,
        prepared: PreparedTurn,
        presenter: Presenter,
    ) -> AsyncGenerator[str]:
        # NOTE: aclosing, so a client that hangs up releases the conversation lock now
        # rather than whenever the generator happens to be collected
        async with aclosing(self.service.stream(prepared)) as events:
            async for event in events:
                for framed in presenter.present(event):
                    yield encode(framed)
