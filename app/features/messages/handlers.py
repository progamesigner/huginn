import re
from collections.abc import AsyncGenerator
from contextlib import aclosing
from urllib.parse import quote

from starlette.types import Message

from app.contracts.auth import ApiClient
from app.contracts.channels import ATTACHMENT_LIMITS
from lib.guid import GUIDContract
from lib.router import EventSourceResponse, Request, Response, UploadFile
from lib.temporal import ClockContract

from .entities import AttachmentEvent, ErrorEvent, TextEvent
from .exceptions import (
    AttachmentTooLargeError,
    InvalidAttachmentError,
    TurnFailedError,
)
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
    UploadedBlob,
)
from .services import ChatService, PreparedTurn

_HEADERS = {'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
_FILE_LIMIT = max(ATTACHMENT_LIMITS.values())
# NOTE: room for the multipart boundaries and headers around the file itself
_UPLOAD_LIMIT = _FILE_LIMIT + 64 * 1024
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


# NOTE: counts the body as it arrives, so a chunked upload with no declared length is
# cut off at the limit instead of being parsed and spooled to disk in full
def capped(request: Request, limit: int) -> Request:
    received = 0

    async def receive() -> Message:
        nonlocal received
        message = await request.receive()
        received += len(message.get('body', b''))
        if received > limit:
            raise AttachmentTooLargeError(f'uploads are limited to {_FILE_LIMIT} bytes')
        return message

    return Request(request.scope, receive)


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

    # NOTE: the declared length is refused before a byte of the body is read; a body
    # without one is cut off at the same size while it is read
    async def upload(self, request: Request, client: ApiClient) -> UploadedBlob:
        declared = request.headers.get('content-length')
        if (
            declared is not None
            and declared.isdigit()
            and int(declared) > _UPLOAD_LIMIT
        ):
            raise AttachmentTooLargeError(f'uploads are limited to {_FILE_LIMIT} bytes')

        body = capped(request, _UPLOAD_LIMIT)
        async with body.form(max_files=1, max_fields=1) as form:
            file = form.get('file')
            if not isinstance(file, UploadFile):
                raise InvalidAttachmentError('send the file as multipart field "file"')

            data = await file.read(_FILE_LIMIT + 1)
            name = file.filename

        if len(data) > _FILE_LIMIT:
            raise AttachmentTooLargeError(f'uploads are limited to {_FILE_LIMIT} bytes')

        blob = (await self.service.upload(client, data, name)).unwrap()
        return UploadedBlob(
            id=str(blob.id),
            kind=blob.kind,
            mime=blob.mime,
            size=blob.size,
            name=blob.filename,
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
                attachments=ask.attachments,
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
                attachments=ask.latest_attachments(),
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
