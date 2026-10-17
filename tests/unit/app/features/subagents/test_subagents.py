import asyncio
from pathlib import Path
from typing import override

import pytest
from pydantic import SecretStr

from app.features.agents.config import (
    AgentConfig,
    SubagentDefinitionConfig,
    SubagentsConfig,
)
from app.features.agents.contracts import (
    ExecutionContext,
    SubagentDefinition,
    SubagentServiceContract,
    Tool,
    ToolResult,
    TurnStream,
)
from app.features.agents.entities import TurnBudget, TurnOutput, TurnScope
from app.features.agents.execution import execution
from app.features.agents.services import SubagentRunner
from app.features.assets.config import AssetsConfig, FilesystemAssetSourceConfig
from app.features.assets.services import AssetsService
from app.features.subagents.tools import SubagentToolSource
from lib.llm import (
    LLMClient,
    LLMContract,
    ModelRef,
    ProviderConfig,
    TextChunk,
)
from tests.helpers.clock import FakeClock
from tests.helpers.logging import make_logger
from tests.unit.app.features.agents.test_agent_runner import (
    EchoTool,
    FakeLLM,
    StallingLLM,
)


def context(
    *,
    seconds: float = 60,
    concurrency: int = 1,
    tools: dict[str, Tool] | None = None,
) -> ExecutionContext:
    return ExecutionContext(
        deadline=asyncio.get_running_loop().time() + seconds,
        semaphore=asyncio.Semaphore(concurrency),
        scope=TurnScope(
            agent='parent',
            actor='alice',
            channel='bot',
            chat='42',
            handle='new-handle',
            memory_scope='existing-user',
            document_blobs=frozenset({'private'}),
        ),
        model=ModelRef(model='parent', provider='upstream'),
        budget=TurnBudget(min_iteration_budget=0.02, max_iterations=2),
        max_completion_tokens=100,
        tools=tools or {},
    )


class Models(LLMContract):
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @override
    def client_for(self, ref: ModelRef) -> LLMClient | None:
        return self.client if self.provider_for(ref) else None

    @override
    def provider_for(self, ref: ModelRef) -> ProviderConfig | None:
        return (
            ProviderConfig(
                name='upstream',
                adapter='openai',
                api_key=SecretStr('test'),
                models=('parent', 'child'),
            )
            if ref.model in {'parent', 'child'} and ref.provider in {None, 'upstream'}
            else None
        )


class Forbidden(EchoTool):
    execution_policy = 'delegation'


class BriefWrite(EchoTool):
    writes_brief = True


class Document(EchoTool):
    inherits_to_child = False


async def test_real_child_runner_uses_only_task_and_restricts_tools_and_limits() -> (
    None
):
    llm = FakeLLM([[TextChunk(text='child final')]])
    runner = SubagentRunner(Models(llm), FakeClock(), logger=make_logger())
    parent = context(
        tools={
            'echo': EchoTool(),
            'nested': Forbidden(),
            'brief': BriefWrite(),
            'document': Document(),
        },
    )
    result = await runner.execute(
        SubagentDefinition(
            name='review',
            source='config',
            instructions='review carefully',
            model='child',
            max_iterations=99,
            max_completion_tokens=999,
        ),
        'self-contained task',
        parent,
    )
    assert result.text == 'child final'
    request = llm.requests[0]
    assert request.model == 'child'
    assert request.max_completion_tokens == 100
    assert 'private' not in str(request.messages)
    assert 'self-contained task' in str(request.messages)
    assert [
        tool['function']['name']
        for tool in request.tools
        if isinstance(tool, dict) and isinstance(tool['function'], dict)
    ] == ['echo']
    assert len(request.messages) == 2


@pytest.mark.parametrize('reason', ['depth', 'model', 'tool', 'budget'])
async def test_refused_delegation_makes_no_child_llm_call(reason: str) -> None:
    llm = FakeLLM([[TextChunk(text='unused')]])
    parent = context(seconds=0.001 if reason == 'budget' else 60)
    if reason == 'depth':
        parent.depth = 1
    definition = SubagentDefinition(
        name='review',
        source='config',
        instructions='review',
        model='foreign' if reason == 'model' else None,
        tools=('forbidden',) if reason == 'tool' else None,
    )
    with pytest.raises(ValueError, match=r'delegation|model|tools|time'):
        await SubagentRunner(Models(llm), FakeClock(), logger=make_logger()).execute(
            definition,
            'task',
            parent,
        )
    assert not llm.requests


class MeasuringRunner(SubagentServiceContract):
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = 0

    @override
    async def execute(
        self,
        definition: SubagentDefinition,
        task: str,
        context: ExecutionContext,
        stream: TurnStream | None = None,
    ) -> TurnOutput:
        if (
            context.deadline - asyncio.get_running_loop().time()
            < 2 * context.budget.min_iteration_budget
        ):
            raise ValueError('insufficient time')
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.calls.append((definition, task))
        self.started.set()
        try:
            await self.release.wait()
            return TurnOutput(text=definition.instructions)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1


def source(tmp_path: Path, runner: SubagentServiceContract) -> SubagentToolSource:
    settings = AssetsConfig(
        sources={
            'default': FilesystemAssetSourceConfig(kind='filesystem', root=tmp_path),
        },
    )
    assets = AssetsService(settings, None, FakeClock(), logger=make_logger())
    return SubagentToolSource(assets, settings, runner, logger=make_logger())


def configured(allow: tuple[str, ...] = ()) -> AgentConfig:
    return AgentConfig(
        description='parent',
        instructions='help',
        subagents=SubagentsConfig(
            enabled=True,
            allow=allow,
            definitions={
                'review': SubagentDefinitionConfig(
                    description='Review work',
                    instructions='config wins',
                ),
            },
        ),
    )


@pytest.mark.parametrize('concurrency', [1, 2])
async def test_parent_calls_share_a_semaphore_and_cancel_running_and_waiting(
    tmp_path: Path,
    concurrency: int,
) -> None:
    runner = MeasuringRunner()
    parent = context(concurrency=concurrency)
    with execution(parent):
        tool = (
            await source(tmp_path, runner).collect(configured(), parent.scope)
        ).tools[0][1]
        tasks = [
            asyncio.create_task(tool.invoke('{"name":"review","task":"task"}'))
            for _ in range(3)
        ]
        await runner.started.wait()
        await asyncio.sleep(0)
        assert runner.peak == concurrency
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert runner.active == 0
    assert runner.cancelled == concurrency


async def test_allow_config_precedence_fresh_vault_body_and_disabled(
    tmp_path: Path,
) -> None:
    folder = tmp_path / 'Agents/agents/review'
    folder.mkdir(parents=True)
    path = folder / 'PROMPT.md'
    path.write_text('---\ndescription: Review\n---\nvault original')
    runner = MeasuringRunner()
    runner.release.set()
    src = source(tmp_path, runner)
    parent = context()
    with execution(parent):
        assert not (
            await src.collect(AgentConfig(description='off'), parent.scope)
        ).tools
        denied = (await src.collect(configured(('other',)), parent.scope)).tools[0][1]
        assert 'disallowed' in str(
            await denied.invoke('{"name":"review","task":"task"}'),
        )
        tool = (await src.collect(configured(), parent.scope)).tools[0][1]
        assert await tool.invoke('{"name":"review","task":"task"}') == 'config wins'
        agent = configured().model_copy(
            update={'subagents': SubagentsConfig(enabled=True)},
        )
        tool = (await src.collect(agent, parent.scope)).tools[0][1]
        path.write_text('---\ndescription: Review\n---\nvault edited')
        assert await tool.invoke('{"name":"review","task":"task"}') == 'vault edited'


async def test_child_timeout_returns_a_result_and_preserves_parent_time(
    tmp_path: Path,
) -> None:
    llm = StallingLLM('private draft')
    runner = SubagentRunner(Models(llm), FakeClock(), logger=make_logger())
    parent = context(seconds=0.2)
    with execution(parent):
        tool = (
            await source(tmp_path, runner).collect(configured(), parent.scope)
        ).tools[0][1]
        result = await tool.invoke('{"name":"review","task":"task"}')
    assert 'timed out' in str(result) or 'timeout' in str(result)
    assert parent.deadline > asyncio.get_running_loop().time()


async def test_parent_turn_uses_child_result_without_storing_child_conversation(
    tmp_path: Path,
) -> None:
    from app.features.agents.config import DefaultsConfig
    from app.features.agents.contracts import ToolSourceRegistry
    from app.features.agents.entities import RoutingPolicy, TurnPrompt, TurnRequest
    from app.features.agents.router import Router
    from app.features.agents.services import TurnHandler
    from lib.llm import Message, TextSegment, ToolCallChunk
    from tests.helpers.replies import RecordingReply
    from tests.unit.app.features.agents.test_turn_handler import (
        FakeStore,
        FakeTranscript,
    )

    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='delegate',
                    name='agent',
                    arguments='{"name":"review","task":"independent task"}',
                ),
            ],
            [TextChunk(text='child final')],
            [TextChunk(text='parent answer')],
        ],
    )
    models = Models(llm)
    agent = configured().model_copy(
        update={'model': 'parent', 'provider': 'upstream', 'tool_timeout': 0.001},
    )
    history = (Message(role='user', segments=(TextSegment(text='private history'),)),)
    store = FakeStore(FakeTranscript(history))
    reply = RecordingReply()
    runner = SubagentRunner(models, FakeClock(), logger=make_logger())
    handler = TurnHandler(
        models,
        Router({'parent': agent}, models, logger=make_logger()),
        ToolSourceRegistry(lambda: (source(tmp_path, runner),)),
        store,
        FakeClock(),
        defaults=DefaultsConfig(model='parent'),
        logger=make_logger(),
    )
    await handler.handle(
        TurnRequest(
            channel='bot',
            chat='42',
            policy=RoutingPolicy(agent='parent'),
            prompts=(TurnPrompt(text='Delegate a review', actor='alice'),),
        ),
        reply,
    )
    assert len(llm.requests) == 3
    assert 'private history' in str(llm.requests[0].messages)
    assert 'private history' not in str(llm.requests[1].messages)
    assert 'child final' in str(llm.requests[2].messages)
    assert store.transcript.recorded == [(('Delegate a review',), 'parent answer')]
    assert 'child final' not in str(reply)


async def test_queue_and_slow_prompt_load_spend_the_parent_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / 'Agents/agents/review'
    folder.mkdir(parents=True)
    (folder / 'PROMPT.md').write_text(
        '---\nname: review\ndescription: Review\n---\nreview',
    )
    llm = FakeLLM([[TextChunk(text='unused')]])
    runner = SubagentRunner(Models(llm), FakeClock(), logger=make_logger())
    src = source(tmp_path, runner)
    parent = context(seconds=0.08)
    with execution(parent):
        tool = (
            await src.collect(
                configured().model_copy(
                    update={'subagents': SubagentsConfig(enabled=True)},
                ),
                parent.scope,
            )
        ).tools[0][1]
        original = src.assets.load

        async def slow_load(*args, **kwargs):
            await asyncio.sleep(0.05)
            return await original(*args, **kwargs)

        monkeypatch.setattr(src.assets, 'load', slow_load)
        result = await tool.invoke('{"name":"review","task":"task"}')
        assert 'refused' in str(result)
    assert not llm.requests
    queued = context(seconds=0.04)
    await queued.semaphore.acquire()
    with execution(queued):
        tool = (await src.collect(configured(), queued.scope)).tools[0][1]
        assert 'timed out' in str(await tool.invoke('{"name":"review","task":"task"}'))
    assert not llm.requests


async def test_files_a_child_produces_reach_the_parent_turn(tmp_path: Path) -> None:
    from lib.llm import ToolCallChunk
    from tests.unit.app.features.agents.test_agent_runner import ARTIFACT, DrawTool

    llm = FakeLLM(
        [
            [ToolCallChunk(index=0, id='draw', name='draw', arguments='{}')],
            [TextChunk(text='chart attached')],
        ],
    )
    runner = SubagentRunner(Models(llm), FakeClock(), logger=make_logger())
    parent = context(tools={'draw': DrawTool()})
    with execution(parent):
        tool = (
            await source(tmp_path, runner).collect(configured(), parent.scope)
        ).tools[0][1]
        result = await tool.invoke('{"name":"review","task":"draw a chart"}')
    assert isinstance(result, ToolResult)
    assert result.text == 'chart attached'
    assert result.artifacts == (ARTIFACT,)
