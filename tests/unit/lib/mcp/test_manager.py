import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent

from lib.mcp import (
    Connector,
    MCPConnectionError,
    MCPManager,
    MCPServer,
    UnknownMCPServerError,
)
from tests.helpers.logging import make_logger


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
    tried: asyncio.Condition

    def __init__(self, server: FastMCP, failures: int) -> None:
        self.attempts = 0
        self.failures = failures
        self.server = server
        self.tried = asyncio.Condition()

    @asynccontextmanager
    async def __call__(self, _: MCPServer) -> AsyncIterator[ClientSession]:
        async with self.tried:
            self.attempts += 1
            self.tried.notify_all()

        if self.attempts <= self.failures:
            raise ConnectionError('connection refused')

        async with create_connected_server_and_client_session(self.server) as session:
            yield session

    async def wait_for(self, attempts: int) -> None:
        async with self.tried:
            await self.tried.wait_for(lambda: self.attempts >= attempts)


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
        logger=make_logger(),
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
        logger=make_logger(),
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
        logger=make_logger(),
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
        logger=make_logger(),
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
        logger=make_logger(),
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
        logger=make_logger(),
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


# NOTE: without this pass a recovered server stays 'unavailable' until a turn happens to
# ask, which makes the readiness probe lie and makes that turn pay the reconnect
async def test_a_refresh_reopens_a_server_nobody_asked_for() -> None:
    connector = _flaky_connector(_echo_server(), failures=1)
    manager = MCPManager(
        [MCPServer('memory', 'http://memory.test')],
        connector,
        cooldown=0.0,
        logger=make_logger(),
    )

    await manager.start()
    try:
        assert manager.unavailable() == ('memory',)
        await manager.refresh()
        assert manager.unavailable() == ()
    finally:
        await manager.dispose()


async def test_a_server_that_stays_down_is_announced_once_not_once_per_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = MCPManager(
        [MCPServer('memory', 'http://memory.test')],
        _failing_connector,
        cooldown=0.0,
        logger=make_logger(),
    )

    with caplog.at_level('INFO', logger='lib.mcp.manager'):
        await manager.start()
        try:
            await manager.refresh()
            await manager.refresh()
        finally:
            await manager.dispose()

    assert [message for message in caplog.messages if 'unavailable' in message] == [
        'MCP server memory is unavailable; its tools will be absent'
        ' (memory: ConnectionError: connection refused)',
    ]


async def test_a_server_that_comes_back_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connector = _flaky_connector(_echo_server(), failures=1)
    manager = MCPManager(
        [MCPServer('memory', 'http://memory.test')],
        connector,
        cooldown=0.0,
        logger=make_logger(),
    )

    with caplog.at_level('INFO', logger='lib.mcp.manager'):
        await manager.start()
        try:
            await manager.refresh()
        finally:
            await manager.dispose()

    assert any('answered again' in message for message in caplog.messages)


async def test_the_heartbeat_refreshes_without_anyone_asking() -> None:
    connector = _flaky_connector(_echo_server(), failures=1)
    manager = MCPManager(
        [MCPServer('memory', 'http://memory.test')],
        connector,
        cooldown=0.0,
        interval=0.01,
        logger=make_logger(),
    )

    await manager.start()
    try:
        assert connector.attempts == 1
        # NOTE: nothing below touches the manager -- the second attempt is the heartbeat
        await connector.wait_for(2)
    finally:
        await manager.dispose()

    assert manager.heartbeat is None


async def test_degraded_start_logs_the_underlying_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = MCPManager(
        [MCPServer('memory', 'http://memory.test')],
        _group_failing_connector,
        logger=make_logger(),
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
    manager = MCPManager([], _memory_connector(_echo_server()), logger=make_logger())
    await manager.start()
    try:
        with pytest.raises(UnknownMCPServerError):
            await manager.call_tool('missing', 'echo', {})
    finally:
        await manager.dispose()


def test_tool_errors_never_become_successful_text() -> None:
    from lib.mcp import MCPError
    from lib.mcp.manager import _render

    with pytest.raises(MCPError, match='returned an error'):
        _render(
            CallToolResult(
                isError=True,
                content=[TextContent(type='text', text='private')],
            ),
        )


def test_structured_only_tools_retain_their_payload() -> None:
    import json

    from lib.mcp.manager import _render

    assert json.loads(
        _render(
            CallToolResult(content=[], structuredContent={'rendered': 'whole context'}),
        ),
    ) == {'rendered': 'whole context'}
