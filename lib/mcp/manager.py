import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import override

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent
from pydantic import JsonValue

from lib.http import HTTPClientContract

from .contracts import MCPClientContract
from .exceptions import MCPConnectionError, UnknownMCPServerError
from .types import MCPServer, MCPToolInfo

logger = logging.getLogger(__name__)

type Connector = Callable[[MCPServer], AbstractAsyncContextManager[ClientSession]]

_RECONNECT_COOLDOWN = 30.0


def streamable_http_connector(http: HTTPClientContract) -> Connector:
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

    def __init__(
        self,
        server: MCPServer,
        connector: Connector,
        cooldown: float = _RECONNECT_COOLDOWN,
    ) -> None:
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
            logger.info('Opening a session to MCP server %s', self.server.name)
            try:
                await self._launch()
            except MCPConnectionError:
                self.retry_at = loop.time() + self.cooldown
                raise

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

    def __init__(
        self,
        servers: Sequence[MCPServer],
        connector: Connector,
        cooldown: float = _RECONNECT_COOLDOWN,
    ) -> None:
        self.connector = connector
        self.servers = tuple(servers)
        self.connections = {
            server.name: _Connection(server, connector, cooldown)
            for server in self.servers
        }

    async def dispose(self) -> None:
        await asyncio.gather(
            *(connection.aclose() for connection in self.connections.values()),
        )
        self.connections.clear()

    async def start(self) -> None:
        await asyncio.gather(
            *(self._connect(connection) for connection in self.connections.values()),
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

    async def _connect(self, connection: _Connection) -> None:
        try:
            await connection.ensure()
        except MCPConnectionError as error:
            # NOTE: not fail to boot because a tool server is briefly unreachable
            logger.warning(
                'MCP server %s is unavailable; its tools will be absent (%s)',
                connection.server.name,
                error,
            )

    def _connection(self, server: str) -> _Connection:
        if server not in self.connections:
            raise UnknownMCPServerError(f'mcp server {server!r} is not configured')
        return self.connections[server]
