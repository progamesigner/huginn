from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from httpx import URL, AsyncClient, AsyncHTTPTransport, Limits, Timeout

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_CONNECTIONS = 100
_DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 20


class HTTPClientManager:
    limits: Limits
    timeout: Timeout
    transport: AsyncHTTPTransport

    def __init__(
        self,
        timeout: float = _DEFAULT_TIMEOUT,
        max_connections: int = _DEFAULT_MAX_CONNECTIONS,
        max_keepalive_connections: int = _DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
    ) -> None:
        self.limits = Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )
        self.timeout = Timeout(timeout)
        self.transport = AsyncHTTPTransport(limits=self.limits)

    @asynccontextmanager
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
