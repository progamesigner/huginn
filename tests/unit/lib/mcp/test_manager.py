from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from lib.mcp import (
    Connector,
    MCPConnectionError,
    MCPManager,
    MCPServer,
    UnknownMCPServerError,
)


def _echo_server() -> FastMCP:
    server = FastMCP('test')

    @server.tool()
    def echo(text: str) -> str:
        """Echo the text back to the caller."""
        return f'echoed: {text}'

    return server


def _memory_connector(server: FastMCP) -> Connector:
    @asynccontextmanager
    async def connect(_: MCPServer) -> AsyncIterator[ClientSession]:
        async with create_connected_server_and_client_session(server) as session:
            yield session

    return connect


class _FlakyConnector:
    attempts: int
    failures: int
    server: FastMCP

    def __init__(self, server: FastMCP, failures: int) -> None:
        self.attempts = 0
        self.failures = failures
        self.server = server

    @asynccontextmanager
    async def __call__(self, _: MCPServer) -> AsyncIterator[ClientSession]:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise ConnectionError('connection refused')

        async with create_connected_server_and_client_session(self.server) as session:
            yield session


def _flaky_connector(server: FastMCP, failures: int) -> _FlakyConnector:
    return _FlakyConnector(server, failures)


async def _kill_session(manager: MCPManager) -> None:
    # NOTE: what a server restart looks like from here -- the serving task unwinds and
    # leaves the connection sessionless, exactly as the live failure did
    connection = manager.connections['memory']
    await connection.aclose()
    assert not connection.connected


@asynccontextmanager
async def _failing_connector(_: MCPServer) -> AsyncIterator[ClientSession]:
    raise ConnectionError('connection refused')
    yield  # pragma: no cover


@asynccontextmanager
async def _group_failing_connector(_: MCPServer) -> AsyncIterator[ClientSession]:
    raise ExceptionGroup(
        'unhandled errors in a TaskGroup',
        [ConnectionError('server responded with 404')],
    )
    yield  # pragma: no cover


async def test_list_tools_maps_to_our_types() -> None:
    manager = MCPManager(
        [MCPServer('memory', 'http://memory.test')],
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
    manager = MCPManager(
        [MCPServer('memory', 'http://memory.test')],
        _memory_connector(_echo_server()),
    )
    await manager.start()
    try:
        output = await manager.call_tool('memory', 'echo', {'text': 'hi'})
    finally:
        await manager.dispose()

    assert output == 'echoed: hi'


async def test_unreachable_server_degrades_without_raising() -> None:
    manager = MCPManager(
        [MCPServer('memory', 'http://memory.test')],
        _failing_connector,
    )

    await manager.start()
    try:
        assert manager.unavailable() == ('memory',)
        with pytest.raises(MCPConnectionError):
            await manager.list_tools('memory')
    finally:
        await manager.dispose()


# NOTE: the manager used to keep only the servers that answered at boot, so a server that
# was down for the first second of the process stayed unreachable until a restart
async def test_a_server_that_was_down_at_boot_is_used_once_it_answers() -> None:
    connector = _flaky_connector(_echo_server(), failures=1)
    manager = MCPManager(
        [MCPServer('memory', 'http://memory.test')],
        connector,
        cooldown=0.0,
    )

    await manager.start()
    try:
        assert manager.unavailable() == ('memory',)
        tools = await manager.list_tools('memory')
        assert [tool.name for tool in tools] == ['echo']
        assert manager.unavailable() == ()
    finally:
        await manager.dispose()


# NOTE: the live failure -- the boot session died mid-life and every later turn raised
# 'muninn: not connected' until the process was restarted
async def test_a_session_that_died_is_reopened_on_the_next_call() -> None:
    manager = MCPManager(
        [MCPServer('memory', 'http://memory.test')],
        _memory_connector(_echo_server()),
        cooldown=0.0,
    )
    await manager.start()
    try:
        await _kill_session(manager)
        assert manager.unavailable() == ('memory',)

        assert await manager.call_tool('memory', 'echo', {'text': 'hi'}) == 'echoed: hi'
        assert manager.unavailable() == ()
    finally:
        await manager.dispose()


async def test_a_dead_server_is_not_retried_until_the_cooldown_lapses() -> None:
    connector = _flaky_connector(_echo_server(), failures=1)
    manager = MCPManager(
        [MCPServer('memory', 'http://memory.test')],
        connector,
        cooldown=30.0,
    )

    await manager.start()
    try:
        # NOTE: one attempt at boot, and no second one while the cooldown holds -- a
        # server that is genuinely down must not cost every turn a connect timeout
        with pytest.raises(MCPConnectionError):
            await manager.list_tools('memory')

        assert connector.attempts == 1
    finally:
        await manager.dispose()


async def test_degraded_start_logs_the_underlying_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = MCPManager(
        [MCPServer('memory', 'http://memory.test')],
        _group_failing_connector,
    )

    with caplog.at_level('WARNING', logger='lib.mcp.manager'):
        await manager.start()
    try:
        assert manager.unavailable() == ('memory',)
        assert any(
            'server responded with 404' in message for message in caplog.messages
        )
    finally:
        await manager.dispose()


async def test_call_unknown_server_raises() -> None:
    manager = MCPManager([], _memory_connector(_echo_server()))
    await manager.start()
    try:
        with pytest.raises(UnknownMCPServerError):
            await manager.call_tool('missing', 'echo', {})
    finally:
        await manager.dispose()
