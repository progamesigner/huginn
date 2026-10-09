import re
from collections.abc import AsyncGenerator
from contextlib import aclosing
from urllib.parse import quote

from app.contracts.auth import ApiClient
from lib.guid import GUIDContract
from lib.router import EventSourceResponse, Response
from lib.temporal import ClockContract

from .entities import AttachmentEvent, ErrorEvent, TextEvent
from .exceptions import TurnFailedError
from .presenters import (
    AnthropicPresenter,
    NativePresenter,
    OpenAIPresenter,
    Presenter,
    attachment_link,
    encode,
)
from .schemas import (
    ChatRequest,
    CompatibleMessage,
    CompatibleRequest,
    Completion,
    CompletionChoice,
    CompletionMessage,
    MessageContent,
)
from .services import ChatService, PreparedTurn

_HEADERS = {'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
_FALLBACK_NAME = 'download'
_UNSAFE_NAME = re.compile(r'[^A-Za-z0-9._-]+')
_CONTROL = re.compile(r'[\x00-\x1f\x7f/\\]+')
_COMPLETION_PREFIX = 'chatcmpl'
_MESSAGE_PREFIX = 'msg'


# NOTE: the name came from a model or a user, and it lands in a header -- the plain form
# keeps only what no client misreads, the encoded form carries the rest escaped
def disposition(name: str | None) -> str:
    cleaned = _CONTROL.sub('_', name or '').strip(' .') or _FALLBACK_NAME
    plain = _UNSAFE_NAME.sub('_', cleaned).strip('_')
    if not plain.strip('.'):
        plain = _FALLBACK_NAME
    elif plain.startswith('.'):
        plain = _FALLBACK_NAME + plain
    return (
        f'attachment; filename="{plain}"; filename*=UTF-8\'\'{quote(cleaned, safe="")}'
    )


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

    async def anthropic(
        self,
        ask: CompatibleRequest,
        client: ApiClient,
    ) -> EventSourceResponse | CompatibleMessage:
        prepared = await self._prepare(ask, client)
        identifier = self._identify(_MESSAGE_PREFIX)
        if not ask.stream:
            return CompatibleMessage(
                id=identifier,
                model=ask.model,
                content=(MessageContent(text=await self._answer(prepared)),),
            )

        return self._streaming(
            prepared,
            AnthropicPresenter(identifier, ask.model),
        )

    async def download(self, blob: str, client: ApiClient) -> Response:
        found = (await self.service.download(client, blob)).unwrap()
        return Response(
            content=found.data,
            media_type=found.mime,
            headers={
                'Content-Disposition': disposition(found.filename),
                'Cache-Control': 'private, no-store',
                'X-Content-Type-Options': 'nosniff',
            },
        )

    async def native(self, ask: ChatRequest, client: ApiClient) -> EventSourceResponse:
        # NOTE: the HTTP boundary turns an expected failure into the error it answers
        prepared = (
            await self.service.prepare(
                client,
                ask.message,
                conversation=ask.conversation,
                agent=ask.agent,
                max_completion_tokens=ask.max_completion_tokens,
            )
        ).unwrap()

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
                    case AttachmentEvent():
                        fragments.append(attachment_link(event))
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
        return (
            await self.service.prepare(
                client,
                ask.latest(),
                conversation=ask.thread(),
                agent=ask.model if ask.model in client.agents else None,
                max_completion_tokens=ask.max_completion_tokens,
            )
        ).unwrap()

    def _streaming(
        self,
        prepared: PreparedTurn,
        presenter: Presenter,
    ) -> EventSourceResponse:
        return EventSourceResponse(
            self.frames(prepared, presenter),
            headers=_HEADERS,
        )
