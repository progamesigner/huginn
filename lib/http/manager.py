from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import override

from httpx2 import (
    URL,
    AsyncBaseTransport,
    AsyncClient,
    AsyncHTTPTransport,
    Limits,
    Timeout,
)

from .contracts import HTTPClientContract

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_CONNECTIONS = 100
_DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 20
_DEFAULT_KEEPALIVE_EXPIRY = 300.0
_DEFAULT_HTTP2 = True


class HTTPClientManager(HTTPClientContract):
    limits: Limits
    timeout: Timeout
    transport: AsyncBaseTransport

    def __init__(
        self,
        timeout: float = _DEFAULT_TIMEOUT,
        max_connections: int = _DEFAULT_MAX_CONNECTIONS,
        max_keepalive_connections: int = _DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry: float = _DEFAULT_KEEPALIVE_EXPIRY,
        http2: bool = _DEFAULT_HTTP2,
        transport: AsyncBaseTransport | None = None,
    ) -> None:
        self.limits = Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
        )
        self.timeout = Timeout(timeout)
        # NOTE: a given transport replaces the pool, which is how tests answer requests
        self.transport = transport or AsyncHTTPTransport(
            limits=self.limits,
            http2=http2,
        )

    @asynccontextmanager
    @override
    async def connect(
        self,
        base_url: URL | str = '',
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[AsyncClient]:
        # NOTE: clients are cheap facades over the shared pool
        yield AsyncClient(
            base_url=base_url,
            headers=dict(headers or {}),
            timeout=self.timeout,
            transport=self.transport,
        )

    async def dispose(self) -> None:
        await self.transport.aclose()
