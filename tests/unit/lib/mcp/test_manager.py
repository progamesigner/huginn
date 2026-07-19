from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from lib.mcp import Connector, McpManager, McpServer, UnknownMcpServerError


def _echo_server() -> FastMCP:
    server = FastMCP('test')

    @server.tool()
    def echo(text: str) -> str:
        """Echo the text back to the caller."""
        return f'echoed: {text}'

    return server


def _memory_connector(server: FastMCP) -> Connector:
    @asynccontextmanager
    async def connect(_: McpServer) -> AsyncIterator[ClientSession]:
        async with create_connected_server_and_client_session(server) as session:
            yield session

    return connect


@asynccontextmanager
async def _failing_connector(_: McpServer) -> AsyncIterator[ClientSession]:
    raise ConnectionError('connection refused')
    yield  # pragma: no cover


async def test_list_tools_maps_to_our_types() -> None:
    manager = McpManager(
        [McpServer('memory', 'http://memory.test')],
        _memory_connector(_echo_server()),
    )
    await manager.start()
    try:
        tools = await manager.list_tools('memory')
    finally:
        await manager.dispose()

    echo = next(tool for tool in tools if tool.name == 'echo')
    assert echo.description == 'Echo the text back to the caller.'
    assert isinstance(echo.input_schema, dict)
    assert 'text' in echo.input_schema['properties']


async def test_call_tool_returns_rendered_text() -> None:
    manager = McpManager(
        [McpServer('memory', 'http://memory.test')],
        _memory_connector(_echo_server()),
    )
    await manager.start()
    try:
        output = await manager.call_tool('memory', 'echo', {'text': 'hi'})
    finally:
        await manager.dispose()

    assert output == 'echoed: hi'


async def test_unreachable_server_degrades_without_raising() -> None:
    manager = McpManager(
        [McpServer('memory', 'http://memory.test')], _failing_connector
    )

    await manager.start()
    try:
        assert manager.connections == {}
        assert manager.unavailable == ['memory']
        with pytest.raises(UnknownMcpServerError):
            await manager.list_tools('memory')
    finally:
        await manager.dispose()


async def test_call_unknown_server_raises() -> None:
    manager = McpManager([], _memory_connector(_echo_server()))
    await manager.start()
    try:
        with pytest.raises(UnknownMcpServerError):
            await manager.call_tool('missing', 'echo', {})
    finally:
        await manager.dispose()
