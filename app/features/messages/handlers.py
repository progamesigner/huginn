from collections.abc import AsyncGenerator
from contextlib import aclosing

from app.contracts.auth import ApiClient
from lib.guid import GUIDContract
from lib.router import EventSourceResponse
from lib.temporal import ClockContract

from .exceptions import TurnFailedError
from .presenters import NativePresenter, OpenAIPresenter, Presenter, encode
from .services import ChatService, PreparedTurn
from .types import (
    ChatRequest,
    CompatibleRequest,
    Completion,
    CompletionChoice,
    CompletionMessage,
    ErrorEvent,
    TextEvent,
)

_HEADERS = {'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
_COMPLETION_PREFIX = 'chatcmpl'


class ChatHandlers:
    clock: ClockContract
    guid: GUIDContract
    service: ChatService

    def __init__(
        self,
        service: ChatService,
        guid: GUIDContract,
        clock: ClockContract,
    ) -> None:
        self.clock = clock
        self.guid = guid
        self.service = service

    async def native(self, ask: ChatRequest, client: ApiClient) -> EventSourceResponse:
        prepared = await self.service.prepare(
            client,
            ask.message,
            conversation=ask.conversation,
            agent=ask.agent,
        )

        return self._streaming(prepared, NativePresenter())

    async def openai(
        self,
        ask: CompatibleRequest,
        client: ApiClient,
    ) -> EventSourceResponse | Completion:
        prepared = await self._prepare(ask, client)
        identifier = self._identify(_COMPLETION_PREFIX)
        created = int(self.clock.now().timestamp())
        if not ask.stream:
            return Completion(
                id=identifier,
                created=created,
                model=ask.model,
                choices=(
                    CompletionChoice(
                        message=CompletionMessage(
                            content=await self._answer(prepared),
                        ),
                    ),
                ),
            )

        return self._streaming(
            prepared,
            OpenAIPresenter(identifier, created, ask.model),
        )

    async def frames(
        self,
        prepared: PreparedTurn,
        presenter: Presenter,
    ) -> AsyncGenerator[str]:
        async with aclosing(self.service.stream(prepared)) as events:
            async for event in events:
                for framed in presenter.present(event):
                    yield encode(framed)

    async def _answer(self, prepared: PreparedTurn) -> str:
        fragments = []
        async with aclosing(self.service.stream(prepared)) as events:
            async for event in events:
                match event:
                    case TextEvent(text=text):
                        fragments.append(text)
                    case ErrorEvent(message=message):
                        raise TurnFailedError(message)
                    case _:
                        continue

        return ''.join(fragments)

    def _identify(self, prefix: str) -> str:
        return f'{prefix}-{self.guid.generate().hex}'

    async def _prepare(
        self,
        ask: CompatibleRequest,
        client: ApiClient,
    ) -> PreparedTurn:
        return await self.service.prepare(
            client,
            ask.latest(),
            conversation=ask.thread(),
            agent=ask.model if ask.model in client.agents else None,
        )

    def _streaming(
        self,
        prepared: PreparedTurn,
        presenter: Presenter,
    ) -> EventSourceResponse:
        return EventSourceResponse(
            self.frames(prepared, presenter),
            headers=_HEADERS,
        )
