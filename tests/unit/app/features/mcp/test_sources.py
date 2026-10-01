from collections.abc import Mapping
from typing import Any, cast, override

import pytest
from pydantic import JsonValue

from app.features.agents.config import AgentConfig, MCPRef
from app.features.agents.entities import (
    RoutingPolicy,
    TurnPrompt,
    TurnRequest,
    TurnScope,
)
from app.features.mcp.sources import MCPToolSource
from lib.llm import CompletionChunk, FinishChunk, TextChunk, ToolCallChunk
from lib.mcp import (
    MCPClientContract,
    MCPConnectionError,
    MCPToolInfo,
    UnknownMCPServerError,
)
from tests.helpers.logging import make_logger
from tests.helpers.replies import RecordingReply
from tests.unit.app.features.agents.test_agent_runner import FakeLLM
from tests.unit.app.features.agents.test_turn_handler import (
    FakeStore,
    configured_handler,
)


class FakeClient(MCPClientContract):
    disconnected: tuple[str, ...]
    tools: dict[str, tuple[MCPToolInfo, ...]]
    calls: list[tuple[str, str, Mapping[str, JsonValue]]]

    def __init__(
        self,
        tools: dict[str, tuple[MCPToolInfo, ...]],
        disconnected: tuple[str, ...] = (),
    ) -> None:
        self.disconnected = disconnected
        self.tools = tools
        self.calls = []

    @override
    async def list_tools(self, server: str) -> tuple[MCPToolInfo, ...]:
        if server in self.disconnected:
            raise MCPConnectionError(f'{server}: not connected')

        if server not in self.tools:
            raise UnknownMCPServerError(server)

        return self.tools[server]

    @override
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        self.calls.append((server, tool, arguments))
        return 'called'

    @override
    def unavailable(self) -> tuple[str, ...]:
        return ()


_SCOPE = TurnScope(actor='user:1', agent='test', channel='test', chat='1')


def _info(name: str) -> MCPToolInfo:
    return MCPToolInfo(name=name, description='', input_schema={'type': 'object'})


def _agent(*refs: MCPRef) -> AgentConfig:
    return AgentConfig(description='x', instructions='y', mcp=refs)


async def test_collect_namespaces_and_applies_allowlist() -> None:
    client = FakeClient({'memory': (_info('recall'), _info('forget'))})
    agent = _agent(MCPRef(server='memory', allow=('recall',), inject={'user': 'swag'}))

    harvest = await MCPToolSource(client, logger=make_logger()).collect(agent, _SCOPE)

    assert [name for name, _ in harvest.tools] == ['mcp__memory__recall']


async def test_collect_binds_the_turn_scope_into_tool_arguments() -> None:
    info = MCPToolInfo(
        name='recall',
        description='',
        input_schema={
            'type': 'object',
            'properties': {'query': {'type': 'string'}, 'user': {'type': 'string'}},
            'required': ['query', 'user'],
        },
    )
    client = FakeClient({'memory': (info,)})
    agent = _agent(
        MCPRef(server='memory', allow=('recall',), bind={'user': 'handle'}),
    )
    scope = TurnScope(
        actor='user:1',
        agent='test',
        channel='test',
        chat='1',
        handle='alice',
    )

    harvest = await MCPToolSource(client, logger=make_logger()).collect(agent, scope)
    tool = harvest.tools[0][1]
    await tool.invoke('{"query":"weather"}')

    schema = cast(dict[str, Any], tool.schema())
    assert 'user' not in schema['properties']
    assert client.calls == [('memory', 'recall', {'query': 'weather', 'user': 'alice'})]


async def test_collect_withholds_a_server_when_a_binding_is_missing() -> None:
    client = FakeClient({'memory': (_info('recall'),)})
    agent = _agent(MCPRef(server='memory', bind={'user': 'handle'}))

    harvest = await MCPToolSource(client, logger=make_logger()).collect(agent, _SCOPE)

    assert harvest.tools == ()


async def test_collect_skips_unavailable_servers() -> None:
    client = FakeClient({'memory': (_info('recall'),)})
    agent = _agent(
        MCPRef(server='stock', allow=('quote',)),
        MCPRef(server='memory', allow=('recall',)),
    )

    harvest = await MCPToolSource(client, logger=make_logger()).collect(agent, _SCOPE)

    assert [name for name, _ in harvest.tools] == ['mcp__memory__recall']
    assert harvest.unreachable == ('stock',)


# NOTE: the live failure -- a configured server whose session had died raised
# MCPConnectionError, which this source did not catch, and every turn died on tool
# assembly before the model was ever asked anything
async def test_collect_skips_a_server_that_lost_its_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeClient({'memory': (_info('recall'),)}, disconnected=('muninn',))
    agent = _agent(
        MCPRef(server='muninn', allow=('recall_memory_notes',)),
        MCPRef(server='memory', allow=('recall',)),
    )

    with caplog.at_level('WARNING'):
        harvest = await MCPToolSource(client, logger=make_logger()).collect(
            agent,
            _SCOPE,
        )

    assert [name for name, _ in harvest.tools] == ['mcp__memory__recall']
    assert harvest.unreachable == ('muninn',)
    assert any('muninn' in message for message in caplog.messages)


def _recalling() -> list[list[CompletionChunk]]:
    return [
        [
            ToolCallChunk(
                index=0,
                id='call-1',
                name='mcp__memory__recall',
                arguments='{"query":"plans"}',
            ),
            FinishChunk(reason='tool_calls'),
        ],
        [TextChunk(text='done'), FinishChunk(reason='stop')],
    ]


def _turn(account: str, handle: str) -> TurnRequest:
    return TurnRequest(
        channel='support_bot',
        chat='4242',
        policy=RoutingPolicy(agent='scout'),
        prompts=(
            TurnPrompt(
                text='what did I plan?',
                actor=f'user:{account}',
                account=account,
                handle=handle,
            ),
        ),
    )


async def test_two_accounts_on_one_bot_each_reach_their_own_memory_scope() -> None:
    info = MCPToolInfo(
        name='recall',
        description='',
        input_schema={
            'type': 'object',
            'properties': {'query': {'type': 'string'}, 'user': {'type': 'string'}},
        },
    )
    client = FakeClient({'memory': (info,)})
    handler = configured_handler(
        FakeLLM([*_recalling(), *_recalling()]),
        FakeStore(),
        MCPToolSource(client, logger=make_logger()),
        mcp=(MCPRef(server='memory', allow=('recall',), bind={'user': 'handle'}),),
    )

    await handler.handle(_turn('account-1', 'alice'), RecordingReply())
    await handler.handle(_turn('account-2', 'bob'), RecordingReply())

    assert [arguments['user'] for _, _, arguments in client.calls] == ['alice', 'bob']
