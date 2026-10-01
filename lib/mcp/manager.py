import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import override

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent
from pydantic import JsonValue
from structlog.stdlib import BoundLogger

from .contracts import HTTPConnector, MCPClientContract
from .exceptions import MCPConnectionError, UnknownMCPServerError
from .types import MCPServer, MCPToolInfo

type Connector = Callable[[MCPServer], AbstractAsyncContextManager[ClientSession]]

_RECONNECT_COOLDOWN = 30.0
_HEARTBEAT_INTERVAL = 30.0


# NOTE: a bindable holder, since a bare callable type cannot key a container binding
@dataclass(frozen=True)
class MCPTransport:
    connector: Connector


def streamable_http_connector(http: HTTPConnector) -> Connector:
    @asynccontextmanager
    async def connect(server: MCPServer) -> AsyncIterator[ClientSession]:
        async with http.connect(headers=server.headers) as client:
            async with streamable_http_client(server.url, http_client=client) as (
                read,
                write,
                _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    return connect


def _describe(error: BaseException | None) -> str:
    # NOTE: anyio task groups bury the real failure inside ExceptionGroup
    if error is None:
        return 'connection closed unexpectedly'
    if isinstance(error, BaseExceptionGroup):
        return '; '.join(_describe(exception) for exception in error.exceptions)
    return f'{type(error).__name__}: {error}'


def _render(result: CallToolResult) -> str:
    return '\n'.join(
        block.text for block in result.content if isinstance(block, TextContent)
    )


class _Connection:
    server: MCPServer
    connector: Connector
    cooldown: float
    session: ClientSession | None
    task: asyncio.Task[None] | None
    ready: asyncio.Event
    stop: asyncio.Event
    error: BaseException | None
    lock: asyncio.Lock
    retry_at: float
    announced: bool
    logger: BoundLogger

    def __init__(
        self,
        server: MCPServer,
        connector: Connector,
        logger: BoundLogger,
        cooldown: float = _RECONNECT_COOLDOWN,
    ) -> None:
        self.logger = logger
        self.server = server
        self.connector = connector
        self.cooldown = cooldown
        self.session = None
        self.task = None
        self.ready = asyncio.Event()
        self.stop = asyncio.Event()
        self.error = None
        self.lock = asyncio.Lock()
        self.retry_at = 0.0
        self.announced = False

    @property
    def connected(self) -> bool:
        return self.session is not None

    async def aclose(self) -> None:
        self.stop.set()
        if self.task is not None:
            await self.task
            self.task = None

    async def ensure(self) -> None:
        if self.connected:
            return

        async with self.lock:
            if self.connected:
                return

            loop = asyncio.get_running_loop()
            if loop.time() < self.retry_at:
                raise MCPConnectionError(f'{self.server.name}: not connected')

            await self.aclose()
            try:
                await self._launch()
            except MCPConnectionError as error:
                self.retry_at = loop.time() + self.cooldown
                self._announce(error)
                raise

            self._recovered()

    async def call_tool(self, tool: str, arguments: Mapping[str, JsonValue]) -> str:
        result = await self._require().call_tool(tool, dict(arguments))
        return _render(result)

    async def list_tools(self) -> tuple[MCPToolInfo, ...]:
        result = await self._require().list_tools()
        return tuple(
            MCPToolInfo(
                name=tool.name,
                description=tool.description or '',
                input_schema=tool.inputSchema,
            )
            for tool in result.tools
        )

    def _announce(self, error: MCPConnectionError) -> None:
        if self.announced:
            return

        self.announced = True
        self.logger.warning(
            'MCP server %s is unavailable; its tools will be absent (%s)',
            self.server.name,
            error,
        )

    def _recovered(self) -> None:
        if not self.announced:
            return

        self.announced = False
        self.logger.info('MCP server %s answered again', self.server.name)

    async def _launch(self) -> None:
        self.error = None
        self.ready = asyncio.Event()
        self.stop = asyncio.Event()
        self.task = asyncio.create_task(self._serve())
        await self.ready.wait()
        if self.session is None:
            raise MCPConnectionError(
                f'{self.server.name}: {_describe(self.error)}',
            ) from self.error

    def _require(self) -> ClientSession:
        if self.session is None:
            raise MCPConnectionError(f'{self.server.name}: not connected')
        return self.session

    async def _serve(self) -> None:
        try:
            async with self.connector(self.server) as session:
                self.session = session
                self.ready.set()
                await self.stop.wait()
        except Exception as error:
            self.error = error
        finally:
            self.session = None
            self.ready.set()


class MCPManager(MCPClientContract):
    connections: dict[str, _Connection]
    connector: Connector
    servers: tuple[MCPServer, ...]
    interval: float
    heartbeat: asyncio.Task[None] | None

    def __init__(
        self,
        servers: Sequence[MCPServer],
        connector: Connector,
        logger: BoundLogger,
        cooldown: float = _RECONNECT_COOLDOWN,
        interval: float = _HEARTBEAT_INTERVAL,
    ) -> None:
        self.connector = connector
        self.servers = tuple(servers)
        self.interval = interval
        self.heartbeat = None
        self.connections = {
            server.name: _Connection(server, connector, logger, cooldown)
            for server in self.servers
        }

    async def dispose(self) -> None:
        await self._stop_beating()
        await asyncio.gather(
            *(connection.aclose() for connection in self.connections.values()),
        )
        self.connections.clear()

    async def start(self) -> None:
        await self.refresh()
        if self.connections:
            self.heartbeat = asyncio.create_task(self._beat())

    async def refresh(self) -> None:
        await asyncio.gather(
            *(self._reach(connection) for connection in self.connections.values()),
        )

    @override
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        connection = self._connection(server)
        await connection.ensure()
        return await connection.call_tool(tool, arguments)

    @override
    async def list_tools(self, server: str) -> tuple[MCPToolInfo, ...]:
        connection = self._connection(server)
        await connection.ensure()
        return await connection.list_tools()

    @override
    def unavailable(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, connection in self.connections.items()
            if not connection.connected
        )

    async def _beat(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            await self.refresh()

    async def _reach(self, connection: _Connection) -> None:
        with suppress(MCPConnectionError):
            await connection.ensure()

    async def _stop_beating(self) -> None:
        if self.heartbeat is None:
            return

        self.heartbeat.cancel()
        await asyncio.gather(self.heartbeat, return_exceptions=True)
        self.heartbeat = None

    def _connection(self, server: str) -> _Connection:
        if server not in self.connections:
            raise UnknownMCPServerError(f'mcp server {server!r} is not configured')
        return self.connections[server]
