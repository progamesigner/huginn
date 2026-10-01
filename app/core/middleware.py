from dataclasses import dataclass, field
from uuid import uuid4

from starlette.types import ASGIApp, Receive, Scope, Send
from structlog.contextvars import bind_contextvars, clear_contextvars


@dataclass(frozen=True)
class RequestContextMiddleware:
    app: ASGIApp
    excludes: frozenset[str] = field(default_factory=frozenset)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] == 'http' and scope['path'] not in self.excludes:
            request_id = uuid4().hex[:12]
            # NOTE: the error envelope reads it off the request, not the log context
            scope.setdefault('state', {})['request_id'] = request_id
            clear_contextvars()
            bind_contextvars(request_id=request_id)

        await self.app(scope, receive, send)
