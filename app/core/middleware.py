from dataclasses import dataclass
from uuid import uuid4

from starlette.types import ASGIApp, Receive, Scope, Send
from structlog.contextvars import bind_contextvars, clear_contextvars

PROBE_PATHS = frozenset({'/healthz', '/readyz'})


@dataclass(frozen=True)
class RequestContextMiddleware:
    app: ASGIApp

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] == 'http' and scope['path'] not in PROBE_PATHS:
            clear_contextvars()
            bind_contextvars(request_id=uuid4().hex[:12])

        await self.app(scope, receive, send)
