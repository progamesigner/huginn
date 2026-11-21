"""Real local Muninn + Huginn brief source/tool loop, with no production fallback.

Run with MUNINN_TEST_BINARY=/absolute/path/to/muninn. Every server uses an empty
explicit environment, loopback binding, and a pytest-owned disposable vault.
The scripted model exercises Huginn's runner without any external model calls.
"""

import asyncio
import json
import os
import socket
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

import httpx2
import pytest

from app.features.agents.config import AgentConfig, BriefConfig
from app.features.agents.contracts import ExecutionContext, ToolHarvest
from app.features.agents.entities import TurnBudget, TurnInput, TurnPrompt, TurnScope
from app.features.agents.execution import execution
from app.features.agents.services import AgentRunner
from app.features.agents.tooling import ToolSet
from app.features.assets.config import AssetsConfig, MCPAssetSourceConfig
from app.features.assets.services import AssetsService
from app.features.briefs.tools import BriefToolSource, UpdateBriefParams
from lib.http import HTTPClientManager
from lib.llm import FinishChunk, ModelCapabilities, ModelRef, TextChunk, ToolCallChunk
from lib.mcp import MCPManager, MCPServer, MCPToolError, streamable_http_connector
from lib.temporal import ClockManager
from tests.helpers.clock import FakeClock
from tests.helpers.logging import make_logger
from tests.helpers.replies import RecordingReply
from tests.unit.app.features.agents.test_agent_runner import FakeLLM

pytestmark = pytest.mark.integration

_REQUEST = 'Please update AGENTS.md with the confirmed decision.'
_SCOPE = {'agent': 'default', 'user': 'default'}


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


@dataclass
class LocalMemory:
    client: MCPManager
    source: BriefToolSource
    root: Path


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[LocalMemory]:
    configured = os.environ.get('MUNINN_TEST_BINARY')
    if not configured:
        pytest.skip('MUNINN_TEST_BINARY is required; no remote service fallback')
    binary = await asyncio.to_thread(Path(configured).resolve, strict=True)
    root = tmp_path / 'vault'
    root.mkdir()
    servers = []
    async with AsyncExitStack() as stack:
        for name, policy in [
            ('first', 'namespaced'),
            ('second', 'namespaced'),
            ('readonly', 'readonly'),
        ]:
            port = _port()
            log = stack.enter_context((tmp_path / f'{name}.log').open('wb'))
            process = await asyncio.create_subprocess_exec(
                str(binary),
                env={
                    'MUNINN_ROOT_DIR': str(root),
                    'MUNINN_AGENTS_DIR': 'Agents',
                    'MUNINN_VFS_SCHEME': '<agent>.<user>',
                    'MUNINN_POLICY': policy,
                    'MUNINN_TRANSPORT': 'http',
                    'MUNINN_HTTP_BIND': f'127.0.0.1:{port}',
                    'MUNINN_RECALL_BACKEND': 'off',
                    'MUNINN_LOG': 'warn',
                },
                cwd=tmp_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=log,
            )

            async def stop(child: asyncio.subprocess.Process = process) -> None:
                if child.returncode is None:
                    child.terminate()
                    try:
                        await asyncio.wait_for(child.wait(), timeout=5)
                    except TimeoutError:
                        child.kill()
                        await child.wait()

            stack.push_async_callback(stop)
            base = f'http://127.0.0.1:{port}'
            async with httpx2.AsyncClient(trust_env=False) as probe:
                async with asyncio.timeout(15):
                    while True:
                        assert process.returncode is None, (
                            'local Muninn exited during startup'
                        )
                        try:
                            if (await probe.get(f'{base}/healthz')).is_success:
                                break
                        except httpx2.TransportError:
                            pass
                        await asyncio.sleep(0.05)
            servers.append(MCPServer(name=name, url=f'{base}/mcp'))
        http = HTTPClientManager()
        stack.push_async_callback(http.dispose)
        client = MCPManager(
            servers,
            streamable_http_connector(http),
            logger=make_logger(),
        )
        stack.push_async_callback(client.dispose)
        await client.start()
        assert client.unavailable() == ()
        settings = AssetsConfig(
            sources={
                name: MCPAssetSourceConfig(
                    server=name,
                    inject={'agent': 'default'},
                    bind={'user': 'memory_scope'},
                )
                for name in ['first', 'second', 'readonly']
            },
        )
        source = BriefToolSource(
            AssetsService(settings, client, FakeClock(), logger=make_logger()),
            settings,
            client,
            logger=make_logger(),
        )
        yield LocalMemory(client, source, root)


def _context(user: str = 'default') -> ExecutionContext:
    return ExecutionContext(
        deadline=asyncio.get_running_loop().time() + 60,
        semaphore=asyncio.Semaphore(2),
        scope=TurnScope(
            agent='assistant',
            actor='user:123',
            channel='test',
            chat='42',
            handle='changed-handle',
            memory_scope=user,
        ),
        model=ModelRef(model='scripted'),
        budget=TurnBudget(min_iteration_budget=0, max_iterations=3),
        max_completion_tokens=100,
        tools={},
        user_messages=(
            TurnPrompt(text=_REQUEST, actor='user:123', message_id='msg-1'),
        ),
    )


async def _collect(
    memory: LocalMemory,
    server: str = 'first',
    *,
    writable: bool = True,
) -> ToolHarvest:
    return await memory.source.collect(
        AgentConfig(
            description='test',
            brief=BriefConfig(enabled=True, writable=writable, source=server),
        ),
        _context().scope,
    )


def _params(content: str) -> UpdateBriefParams:
    return UpdateBriefParams(
        content=content,
        request_message='msg-1',
        request_quote=_REQUEST,
    )


async def test_runner_updates_a_real_scoped_brief_and_the_next_turn_reads_it(
    memory: LocalMemory,
) -> None:
    with execution(_context()):
        harvest = await _collect(memory)
        assert harvest.context == ()
        assert harvest.tools[0][0] == 'update_brief'
        llm = FakeLLM(
            [
                [
                    ToolCallChunk(
                        index=0,
                        id='call-1',
                        name='update_brief',
                        arguments=_params('confirmed locally').model_dump_json(),
                    ),
                    FinishChunk(reason='tool_calls'),
                ],
                [TextChunk(text='Recorded.'), FinishChunk(reason='stop')],
            ],
        )
        tools = ToolSet(dict(harvest.tools), logger=make_logger())
        runner = AgentRunner(llm, tools, ClockManager(), logger=make_logger())
        turn = TurnInput(
            instructions='static prefix',
            prompts=_context().user_messages,
            model='scripted',
            now=FakeClock().now(),
            channel='test',
            capabilities=ModelCapabilities(),
            dynamic_instructions=harvest.context,
            budget=_context().budget,
        )
        result = await runner.run(turn, RecordingReply())
        assert result.text == 'Recorded.'
        assert 'Brief updated.' in str(llm.requests[1].messages)
        following = await _collect(memory)
        assert 'confirmed locally' in following.context[0]
        assert not (await _collect(memory, writable=False)).tools
    physical = memory.root / 'Agents/default.default/AGENTS.default.default.md'
    assert physical.read_text() == 'confirmed locally'
    assert not (memory.root / 'Agents/default.changed-handle').exists()
    with pytest.raises(MCPToolError) as caught:
        await memory.client.call_tool_result(
            'first',
            'read_memory_note',
            {
                **_SCOPE,
                'user': 'other',
                'path': 'Agents/AGENTS.md',
                'include_version': True,
            },
        )
    assert caught.value.code == 'not_found'
    read = json.loads(
        await memory.client.call_tool_result(
            'first',
            'read_memory_note',
            {**_SCOPE, 'path': 'Agents/AGENTS.md', 'include_version': True},
        ),
    )
    with pytest.raises(MCPToolError) as caught:
        await memory.client.call_tool_result(
            'readonly',
            'update_agents_file',
            {**_SCOPE, 'content': 'lost', 'expected_version': read['version']},
        )
    assert caught.value.code == 'write_denied'
    assert physical.read_text() == 'confirmed locally'


@pytest.mark.parametrize('existing', [False, True])
async def test_independent_servers_return_a_conflict_without_overwriting(
    memory: LocalMemory,
    existing: bool,
) -> None:
    if existing:
        await memory.client.call_tool_result(
            'first',
            'update_agents_file',
            {**_SCOPE, 'content': 'initial', 'expected_absent': True},
        )
    with execution(_context()):
        first = (await _collect(memory, 'first')).tools[0][1]
        second = (await _collect(memory, 'second')).tools[0][1]
        outputs = await asyncio.gather(
            first.invoke(_params('first change').model_dump_json()),
            second.invoke(_params('second change').model_dump_json()),
        )
        assert sum(output == 'Brief updated.' for output in outputs) == 1
        assert sum('conflict' in str(output) for output in outputs) == 1
        winner = ['first change', 'second change'][outputs.index('Brief updated.')]
        assert winner in (await _collect(memory)).context[0]
    assert (
        memory.root / 'Agents/default.default/AGENTS.default.default.md'
    ).read_text() == winner
    # The server still enforces root reservation independently of Huginn's bridge.
    with pytest.raises(MCPToolError) as caught:
        await memory.client.call_tool_result(
            'first',
            'write_memory_note',
            {**_SCOPE, 'path': 'Agents/AGENTS.md', 'content': 'lost'},
        )
    assert caught.value.code == 'path_not_permitted'
