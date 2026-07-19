import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import override

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextContent
from pydantic import JsonValue

from .contracts import McpClientContract
from .exceptions import McpConnectionError, UnknownMcpServerError
from .types import McpServer, McpToolInfo

logger = logging.getLogger(__name__)

type Connector = Callable[[McpServer], AbstractAsyncContextManager[ClientSession]]


@asynccontextmanager
async def connect_streamable_http(server: McpServer) -> AsyncIterator[ClientSession]:
    async with streamablehttp_client(server.url, headers=dict(server.headers)) as (
        read,
        write,
        _,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _render(result: CallToolResult) -> str:
    return '\n'.join(
        block.text for block in result.content if isinstance(block, TextContent)
    )


class _Connection:
    server: McpServer
    connector: Connector
    session: ClientSession | None
    task: asyncio.Task[None] | None
    ready: asyncio.Event
    stop: asyncio.Event
    error: BaseException | None

    def __init__(self, server: McpServer, connector: Connector) -> None:
        self.server = server
        self.connector = connector
        self.session = None
        self.task = None
        self.ready = asyncio.Event()
        self.stop = asyncio.Event()
        self.error = None

    async def start(self) -> None:
        # NOTE: the session context must open and close in the same task
        self.task = asyncio.create_task(self._serve())
        await self.ready.wait()
        if self.session is None:
            raise McpConnectionError(f'{self.server.name}: {self.error}')

    async def aclose(self) -> None:
        self.stop.set()
        if self.task is not None:
            await self.task

    async def call_tool(self, tool: str, arguments: Mapping[str, JsonValue]) -> str:
        result = await self._require().call_tool(tool, dict(arguments))
        return _render(result)

    async def list_tools(self) -> tuple[McpToolInfo, ...]:
        result = await self._require().list_tools()
        return tuple(
            McpToolInfo(
                name=tool.name,
                description=tool.description or '',
                input_schema=tool.inputSchema,
            )
            for tool in result.tools
        )

    def _require(self) -> ClientSession:
        if self.session is None:
            raise McpConnectionError(f'{self.server.name}: not connected')
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


class McpManager(McpClientContract):
    connections: dict[str, _Connection]
    connector: Connector
    servers: tuple[McpServer, ...]
    unavailable: list[str]

    def __init__(
        self,
        servers: Sequence[McpServer],
        connector: Connector = connect_streamable_http,
    ) -> None:
        self.connections = {}
        self.connector = connector
        self.servers = tuple(servers)
        self.unavailable = []

    async def dispose(self) -> None:
        await asyncio.gather(
            *(connection.aclose() for connection in self.connections.values())
        )
        self.connections.clear()

    async def start(self) -> None:
        await asyncio.gather(*(self._connect(server) for server in self.servers))

    @override
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        return await self._connection(server).call_tool(tool, arguments)

    @override
    async def list_tools(self, server: str) -> tuple[McpToolInfo, ...]:
        return await self._connection(server).list_tools()

    async def _connect(self, server: McpServer) -> None:
        connection = _Connection(server, self.connector)
        try:
            await connection.start()
        except McpConnectionError as error:
            # NOTE: not fail to boot because a tool server is briefly unreachable
            logger.warning(
                'MCP server %s is unavailable; its tools will be absent (%s)',
                server.name,
                error,
            )
            self.unavailable.append(server.name)
            return

        self.connections[server.name] = connection

    def _connection(self, server: str) -> _Connection:
        if server not in self.connections:
            raise UnknownMcpServerError(f'mcp server {server!r} is not connected')
        return self.connections[server]
