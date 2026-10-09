import asyncio
import json
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Literal, override
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import HttpUrl, JsonValue, ValidationError

from app.core.config.exceptions import ConfigError
from app.core.config.loader import load_config
from app.core.config.registry import SectionRegistry
from app.features.agents.config import AGENTS, AgentConfig
from app.features.agents.contracts import HookRegistry, ToolSourceRegistry
from app.features.agents.entities import (
    RoutingPolicy,
    TurnPrompt,
    TurnRequest,
    TurnScope,
)
from app.features.agents.router import Router
from app.features.agents.services import TurnHandler
from app.features.hooks.config import (
    HOOKS,
    HTTPHookConfig,
    MCPHookConfig,
    ShellHookConfig,
)
from app.features.hooks.sources.configured import ConfiguredHookSource
from app.features.hooks.sources.mcp import _content
from app.features.mcp.config import MCP_SERVERS
from lib.http import HTTPClientContract
from lib.llm import TextSegment
from lib.mcp import MCPClientContract, MCPConnectionError, MCPToolInfo
from tests.helpers.clock import FakeClock
from tests.helpers.logging import make_logger
from tests.helpers.replies import RecordingReply
from tests.unit.app.features.agents.test_turn_handler import (
    DEFAULTS,
    FakeModels,
    FakeStore,
    answering,
)


class Memory(MCPClientContract):
    def __init__(self) -> None:
        self.calls: list[dict[str, JsonValue]] = []
        self.fail = False

    @override
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        assert server == 'memory'
        assert tool == 'load_session_context'
        self.calls.append(dict(arguments))
        if self.fail:
            raise MCPConnectionError('private diagnostic')
        return json.dumps(
            {
                'rendered': f'context:{arguments.get("agent")}:{arguments.get("user")}',
                'missing': [],
            },
        )

    @override
    async def list_tools(self, server: str) -> tuple[MCPToolInfo, ...]:
        return ()

    @override
    def unavailable(self) -> tuple[str, ...]:
        return ()


class HTTP(HTTPClientContract):
    def __init__(
        self,
        status: int = 200,
        *,
        fail: bool = False,
        body: str = 'notice',
    ) -> None:
        self.status = status
        self.fail = fail
        self.body = body
        self.requests: list[httpx.Request] = []

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail:
            raise httpx.ReadTimeout('private diagnostic')
        return httpx.Response(self.status, text=self.body)

    @asynccontextmanager
    @override
    async def connect(
        self,
        base_url: httpx.URL | str = '',
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(self.respond),
            headers=headers,
        ) as client:
            yield client


def scope(handle: str | None = 'alice', agent: str = 'scout') -> TurnScope:
    return TurnScope(
        actor='user:1',
        agent=agent,
        channel='support',
        chat='1',
        handle=handle,
    )


def agent(*hooks: str) -> AgentConfig:
    return AgentConfig(
        description='scout',
        instructions='be helpful',
        model='scout-model',
        hooks=hooks,
    )


def memory_hook(ttl: int | None = 60) -> MCPHookConfig:
    return MCPHookConfig(
        server='memory',
        tool='load_session_context',
        bind={'agent': 'agent', 'user': 'handle'},
        cache_ttl=ttl,
    )


def source(
    config: MCPHookConfig | HTTPHookConfig | ShellHookConfig,
    memory: Memory | None = None,
    http: HTTP | None = None,
    clock: FakeClock | None = None,
) -> ConfiguredHookSource:
    return ConfiguredHookSource(
        {'context': config},
        memory or Memory(),
        http or HTTP(),
        clock or FakeClock(),
        logger=make_logger(),
    )


async def test_cache_isolates_agent_and_user_and_expires() -> None:
    memory = Memory()
    clock = FakeClock()
    hooks = source(memory_hook(), memory, clock=clock)
    for handle, name in [
        ('alice', 'scout'),
        ('bob', 'scout'),
        ('alice', 'writer'),
        ('alice', 'scout'),
    ]:
        harvest = await hooks.collect(agent('context'), scope(handle, name))
        assert harvest.content == f'context:{name}:{handle}'
    assert len(memory.calls) == 3
    clock.advance(timedelta(seconds=60))
    await hooks.collect(agent('context'), scope())
    assert len(memory.calls) == 4
    assert len(hooks.cache) == 1


async def test_no_expiry_and_no_bind_share_the_cache() -> None:
    memory = Memory()
    config = memory_hook(None).model_copy(
        update={'bind': {}, 'arguments': {'agent': 'default', 'user': 'default'}},
    )
    clock = FakeClock()
    hooks = source(config, memory, clock=clock)
    await hooks.collect(agent('context'), scope())
    clock.advance(timedelta(days=100))
    await hooks.collect(agent('context'), scope('bob'))
    assert len(memory.calls) == 1


async def test_concurrent_same_scope_reads_once() -> None:
    memory = Memory()
    hooks = source(memory_hook(), memory)
    await asyncio.gather(*(hooks.collect(agent('context'), scope()) for _ in range(5)))
    assert len(memory.calls) == 1


@pytest.mark.parametrize('handle', [None, 'private value/invalid'])
async def test_missing_or_unsafe_bind_skips_without_disclosing_values(
    handle: str | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    memory = Memory()
    hooks = source(memory_hook(), memory)
    with caplog.at_level('INFO'):
        harvest = await hooks.collect(agent('context'), scope(handle))
    assert harvest.content is None
    assert harvest.unreachable == ()
    assert memory.calls == []
    assert 'user' in caplog.text
    assert 'private value' not in caplog.text


async def test_failure_is_not_cached_and_logs_no_diagnostic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    memory = Memory()
    memory.fail = True
    hooks = source(memory_hook(), memory)
    with caplog.at_level('WARNING'):
        harvest = await hooks.collect(agent('context'), scope())
    assert harvest.unreachable == ('context',)
    assert 'private diagnostic' not in caplog.text
    assert 'alice' not in caplog.text
    memory.fail = False
    assert (
        await hooks.collect(agent('context'), scope())
    ).content == 'context:scout:alice'
    assert len(memory.calls) == 2


async def test_two_accounts_reach_distinct_system_contexts_and_reuse_cache() -> None:
    memory = Memory()
    hooks = source(memory_hook(), memory)
    llm = answering()
    llm.responses *= 4
    models = FakeModels(llm)
    handler = TurnHandler(
        llm=models,
        router=Router(
            {'scout': agent('context')},
            models,
            DEFAULTS,
            logger=make_logger(),
        ),
        sources=ToolSourceRegistry(),
        store=FakeStore(),
        clock=FakeClock(),
        hooks=HookRegistry(lambda: (hooks,)),
        logger=make_logger(),
    )
    for handle in ('alice', 'bob', 'alice'):
        request = TurnRequest(
            channel='support',
            chat='1',
            policy=RoutingPolicy(agent='scout'),
            prompts=(
                TurnPrompt(text='hello', actor='user:1', account=handle, handle=handle),
            ),
        )
        await handler.handle(request, RecordingReply())
        messages = llm.requests[-1].messages
        system = [m for m in messages if m.role == 'system']
        assert len(system) == 1
        segment = system[0].segments[0]
        assert isinstance(segment, TextSegment)
        assert segment.text == f'be helpful\n\ncontext:scout:{handle}'
    assert [call['user'] for call in memory.calls] == ['alice', 'bob']
    memory.fail = True
    request = request.model_copy(
        update={
            'prompts': (
                TurnPrompt(
                    text='hello',
                    actor='user:2',
                    account='charlie',
                    handle='charlie',
                ),
            ),
        },
    )
    reply = RecordingReply()
    await handler.handle(request, reply)
    segment = llm.requests[-1].messages[0].segments[0]
    assert isinstance(segment, TextSegment)
    assert 'Unavailable for this turn' in segment.text


@pytest.mark.parametrize('method', ['GET', 'POST'])
async def test_http_delivers_query_headers_and_truncates(
    method: Literal['GET', 'POST'],
) -> None:
    http = HTTP(body='a' * 70000)
    config = HTTPHookConfig(
        kind='http',
        url=HttpUrl('https://ops.test/context?fixed=keep'),
        method=method,
        headers={'X-Test': 'yes'},
        bind={'user': 'handle'},
    )
    harvest = await source(config, http=http).collect(agent('context'), scope())
    assert harvest.content == 'a' * 65536
    assert http.requests[0].url.params['user'] == 'alice'
    assert http.requests[0].url.params['fixed'] == 'keep'
    assert http.requests[0].headers['X-Test'] == 'yes'
    assert http.requests[0].method == method


@pytest.mark.parametrize(('status', 'fail'), [(503, False), (200, True)])
async def test_http_failure_degrades(status: int, fail: bool) -> None:
    config = HTTPHookConfig(kind='http', url=HttpUrl('https://ops.test/context'))
    harvest = await source(config, http=HTTP(status, fail=fail)).collect(
        agent('context'),
        scope(),
    )
    assert harvest.unreachable == ('context',)


async def test_shell_exec_environment_and_truncation() -> None:
    config = ShellHookConfig(
        kind='shell',
        command=(
            sys.executable,
            '-c',
            "import os; print(os.environ['HUGINN_HOOK_USER']); print('x'*70000)",
        ),
        bind={'user': 'handle'},
    )
    harvest = await source(config).collect(agent('context'), scope())
    assert harvest.content is not None
    assert harvest.content.startswith('alice\n')
    assert len(harvest.content.encode()) == 65536


@pytest.mark.parametrize(
    'command',
    [('/missing/huginn-hook-command',), (sys.executable, '-c', 'raise SystemExit(1)')],
)
async def test_shell_failures_degrade(command: tuple[str, ...]) -> None:
    harvest = await source(ShellHookConfig(kind='shell', command=command)).collect(
        agent('context'),
        scope(),
    )
    assert harvest.unreachable == ('context',)


async def test_shell_timeout_kills_and_reaps(monkeypatch: pytest.MonkeyPatch) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        '-c',
        'import time; time.sleep(10)',
        stdout=asyncio.subprocess.PIPE,
    )
    monkeypatch.setattr(
        asyncio,
        'create_subprocess_exec',
        AsyncMock(return_value=process),
    )
    config = ShellHookConfig(kind='shell', command=('unused',), timeout=1)
    harvest = await source(config).collect(agent('context'), scope())
    assert harvest.unreachable == ('context',)
    assert process.returncode is not None


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ('plain', 'plain'),
        ('"quoted"', 'quoted'),
        ('{"content":"text"}', 'text'),
        ('{"rendered":"whole","missing":[]}', 'whole'),
    ],
)
def test_mcp_text_shapes(raw: str, expected: str) -> None:
    assert _content(raw) == expected


@pytest.mark.parametrize('raw', ['{}', '{"isError":true,"content":"private"}', '[]'])
def test_mcp_invalid_payloads_fail(raw: str) -> None:
    with pytest.raises(ValueError, match='hook'):
        _content(raw)


def test_config_validates_references_and_disjoint_arguments(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text(
        'agents: {scout: {description: x, instructions: y, hooks: [missing]}}',
    )
    with pytest.raises(ConfigError, match=r'agents\.scout\.hooks'):
        load_config(path, SectionRegistry((AGENTS, HOOKS, MCP_SERVERS)))
    path.write_text(
        'agents: {}\nhooks: {context: {kind: mcp, server: absent, tool: load}}',
    )
    with pytest.raises(ConfigError, match=r'hooks\.context\.server'):
        load_config(path, SectionRegistry((AGENTS, HOOKS, MCP_SERVERS)))
    with pytest.raises(ValidationError, match='overlap'):
        MCPHookConfig(
            server='memory',
            tool='load',
            arguments={'user': 'fixed'},
            bind={'user': 'handle'},
        )


async def test_group_with_mixed_accounts_withholds_context() -> None:
    memory = Memory()
    hooks = source(memory_hook(), memory)
    llm = answering()
    models = FakeModels(llm)
    handler = TurnHandler(
        llm=models,
        router=Router(
            {'scout': agent('context')},
            models,
            DEFAULTS,
            logger=make_logger(),
        ),
        sources=ToolSourceRegistry(),
        store=FakeStore(),
        clock=FakeClock(),
        hooks=HookRegistry(lambda: (hooks,)),
        logger=make_logger(),
    )
    request = TurnRequest(
        channel='support',
        chat='1',
        policy=RoutingPolicy(agent='scout'),
        prompts=(
            TurnPrompt(text='hello', actor='user:1', account='alice', handle='alice'),
            TurnPrompt(text='hello', actor='user:2', account='bob', handle='bob'),
        ),
    )
    await handler.handle(request, RecordingReply())
    assert memory.calls == []
    segment = llm.requests[0].messages[0].segments[0]
    assert isinstance(segment, TextSegment)
    assert segment.text == 'be helpful'


def test_root_wiring_reuses_hook_source_across_collections(tmp_path: Path) -> None:
    from app.features.hooks.sources.configured import ConfiguredHookSource
    from app.providers.contributions import (
        AgentContributions,
        CoreContributions,
        HookContributions,
        MCPContributions,
    )
    from tests.helpers.settings import app_with, make_settings

    path = tmp_path / 'config.yaml'
    path.write_text(
        'agents: {scout: {description: x, instructions: y, model: test, hooks: [notice]}}\nhooks: {notice: {kind: shell, command: [uptime]}}',
    )
    app = app_with(make_settings(config_file=path)).build(
        AgentContributions(),
        CoreContributions(),
        HookContributions(),
        MCPContributions(),
    )
    registry = app.resolve(HookRegistry)
    first = registry.sources()
    second = registry.sources()
    assert len(first) == 1
    assert isinstance(first[0], ConfiguredHookSource)
    assert first[0] is second[0]


async def test_unconfigured_agent_does_not_call_hooks() -> None:
    memory = Memory()
    harvest = await source(memory_hook(), memory).collect(agent(), scope())
    assert harvest.content is None
    assert harvest.unreachable == ()
    assert memory.calls == []


async def test_repeated_hook_is_injected_once() -> None:
    memory = Memory()
    harvest = await source(memory_hook(), memory).collect(
        agent('context', 'context'),
        scope(),
    )
    assert harvest.content == 'context:scout:alice'
    assert len(memory.calls) == 1
