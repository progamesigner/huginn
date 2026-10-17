import asyncio
import json
from collections.abc import Mapping
from typing import override

import pytest
from pydantic import JsonValue

from app.features.agents.config import AgentConfig, BriefConfig
from app.features.agents.entities import TurnPrompt
from app.features.agents.execution import execution
from app.features.assets.config import AssetsConfig, MCPAssetSourceConfig
from app.features.assets.services import AssetsService
from app.features.briefs.tools import (
    BriefToolSource,
    UpdateBriefParams,
)
from app.features.mcp.bridge import MCPTool
from lib.mcp import MCPToolInfo
from tests.helpers.clock import FakeClock
from tests.helpers.logging import make_logger
from tests.unit.app.features.assets.test_assets_service import Memory
from tests.unit.app.features.subagents.test_subagents import context


class VersionedMemory(Memory):
    def __init__(self) -> None:
        super().__init__()
        self.docs = {
            'existing-user': ('original brief', 'v1'),
            'bob': ('bob brief', 'v1'),
        }
        self.supported = True

    @override
    async def list_tools(self, server: str) -> tuple[MCPToolInfo, ...]:
        return (
            (
                MCPToolInfo(
                    name='update_agents_file',
                    description='Compare-and-set brief',
                    input_schema={
                        'type': 'object',
                        'properties': {'expected_version': {}, 'expected_absent': {}},
                    },
                ),
            )
            if self.supported
            else ()
        )

    @override
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        self.calls.append((tool, dict(arguments)))
        user = str(arguments['user'])
        previous = self.docs.get(user)
        if tool == 'read_memory_note':
            return json.dumps(
                {'content': previous[0], 'version': previous[1]}
                if previous
                else {'code': 'not_found'},
            )
        if tool == 'update_agents_file':
            expected = arguments.get('expected_version')
            if (
                previous is not None
                and (previous[1] != expected or arguments.get('expected_absent'))
            ) or (previous is None and not arguments.get('expected_absent')):
                return '{"code":"conflict"}'
            version = f'v{int(previous[1][1:]) + 1}' if previous else 'v1'
            self.docs[user] = (str(arguments['content']), version)
            return json.dumps({'version': version})
        return 'called'


def source(memory: VersionedMemory, *, shared: bool = False) -> BriefToolSource:
    settings = AssetsConfig(
        sources={
            'default': (
                MCPAssetSourceConfig(
                    server='memory',
                    inject={'agent': 'default', 'user': 'existing-user'},
                )
                if shared
                else MCPAssetSourceConfig(
                    server='memory',
                    inject={'agent': 'default'},
                    bind={'user': 'memory_scope'},
                )
            ),
        },
    )
    return BriefToolSource(
        AssetsService(settings, memory, FakeClock(), logger=make_logger()),
        settings,
        memory,
        logger=make_logger(),
    )


def agent(writable: bool = True, writers: tuple[str, ...] = ()) -> AgentConfig:
    return AgentConfig(
        description='help',
        brief=BriefConfig(enabled=True, writable=writable, writers=writers),
    )


def params(
    content: str = 'updated brief',
    quote: str = 'Please update AGENTS.md with the confirmed decision.',
) -> UpdateBriefParams:
    return UpdateBriefParams(
        content=content,
        request_message='msg-1',
        request_quote=quote,
    )


async def test_fresh_reads_and_explicit_updates_use_resolved_memory_binding() -> None:
    memory = VersionedMemory()
    src = source(memory)
    parent = context()
    parent.user_messages = (
        TurnPrompt(text=params().request_quote, actor='alice', message_id='msg-1'),
    )
    with execution(parent):
        harvest = await src.collect(agent(), parent.scope)
        assert 'original brief' in harvest.context[0]
        tool = harvest.tools[0][1]
        assert await tool.invoke(params().model_dump_json()) == 'Brief updated.'
        second = await src.collect(agent(), parent.scope)
        assert 'updated brief' in second.context[0]
    assert all(call[1]['user'] == 'existing-user' for call in memory.calls)
    assert memory.docs['bob'][0] == 'bob brief'


@pytest.mark.parametrize(
    ('text', 'origin'),
    [
        ('Let us discuss the brief.', 'user'),
        ('Do not update AGENTS.md.', 'user'),
        ('先不更新 AGENTS.md，只討論內容。', 'user'),  # noqa: RUF001
        ('AGENTS.md 的 update 功能怎麼運作？', 'user'),  # noqa: RUF001
        ('If needed, update AGENTS.md.', 'user'),
        ('Give me a brief update on the deploy.', 'user'),
        ('幫我更新簡報第三頁', 'user'),
        ('Please update AGENTS.md with the confirmed decision.', 'template'),
        ('Please update AGENTS.md with the confirmed decision.', 'delegated'),
    ],
)
async def test_discussion_negation_templates_and_delegation_do_not_authorize_updates(
    text: str,
    origin: str,
) -> None:
    memory = VersionedMemory()
    parent = context()
    parent.user_messages = (
        TurnPrompt.model_validate(
            {'text': text, 'actor': 'alice', 'message_id': 'msg-1', 'origin': origin},
        ),
    )
    with execution(parent):
        tool = (await source(memory).collect(agent(), parent.scope)).tools[0][1]
        assert 'denied' in str(await tool.invoke(params(quote=text).model_dump_json()))
    assert not [call for call in memory.calls if call[0] == 'update_agents_file']


async def test_a_personal_brief_is_updated_only_at_its_owners_request() -> None:
    memory = VersionedMemory()
    parent = context()
    parent.user_messages = (
        TurnPrompt(text=params().request_quote, actor='mallory', message_id='msg-1'),
    )
    with execution(parent):
        tool = (await source(memory).collect(agent(), parent.scope)).tools[0][1]
        assert 'denied' in str(await tool.invoke(params().model_dump_json()))
    assert memory.docs['existing-user'] == ('original brief', 'v1')


async def test_a_shared_brief_without_writers_is_read_only() -> None:
    memory = VersionedMemory()
    parent = context()
    parent.user_messages = (
        TurnPrompt(text=params().request_quote, actor='alice', message_id='msg-1'),
    )
    with execution(parent):
        harvest = await source(memory, shared=True).collect(agent(), parent.scope)
    assert not harvest.tools
    assert 'original brief' in harvest.context[0]
    assert 'read-only' in harvest.context[-1]


async def test_a_shared_brief_accepts_only_configured_writers() -> None:
    memory = VersionedMemory()
    src = source(memory, shared=True)
    shared = agent(writers=('lead',))
    parent = context()
    for handle, expected in (('guest', 'denied'), ('lead', 'Brief updated.')):
        parent.user_messages = (
            TurnPrompt(
                text=params().request_quote,
                actor='alice',
                handle=handle,
                message_id='msg-1',
            ),
        )
        with execution(parent):
            tool = (await src.collect(shared, parent.scope)).tools[0][1]
            assert expected in str(await tool.invoke(params().model_dump_json()))
    assert memory.docs['existing-user'] == ('updated brief', 'v2')


async def test_two_conversations_cannot_silently_overwrite_a_newer_brief() -> None:
    memory = VersionedMemory()
    src = source(memory)
    parent = context()
    parent.user_messages = (
        TurnPrompt(text=params().request_quote, actor='alice', message_id='msg-1'),
    )
    with execution(parent):
        first = (await src.collect(agent(), parent.scope)).tools[0][1]
        second = (await src.collect(agent(), parent.scope)).tools[0][1]
        results = await asyncio.gather(
            first.invoke(params('first change').model_dump_json()),
            second.invoke(params('second change').model_dump_json()),
        )
    assert sum('conflict' in str(result) for result in results) == 1
    assert memory.docs['existing-user'] == ('first change', 'v2')


async def test_missing_brief_uses_expected_absent_and_readonly_has_no_write_tool() -> (
    None
):
    memory = VersionedMemory()
    del memory.docs['existing-user']
    parent = context()
    parent.user_messages = (
        TurnPrompt(text=params().request_quote, actor='alice', message_id='msg-1'),
    )
    with execution(parent):
        src = source(memory)
        readonly = await src.collect(agent(False), parent.scope)
        assert not readonly.context
        assert not readonly.tools
        tool = (await src.collect(agent(), parent.scope)).tools[0][1]
        assert await tool.invoke(params().model_dump_json()) == 'Brief updated.'
    assert memory.calls[-1][1]['expected_absent'] is True
    assert memory.calls[-1][1]['expected_version'] is None


async def test_legacy_server_disables_writes_and_limits_are_rejected_without_truncation() -> (
    None
):
    memory = VersionedMemory()
    parent = context()
    parent.user_messages = (
        TurnPrompt(text=params().request_quote, actor='alice', message_id='msg-1'),
    )
    with execution(parent):
        src = source(memory)
        memory.supported = False
        unavailable = await src.collect(agent(), parent.scope)
        assert not unavailable.tools
        assert not unavailable.unreachable
        assert 'compare-and-set' in unavailable.context[-1]
        memory.supported = True
        tool = (await src.collect(agent(), parent.scope)).tools[0][1]
        for content in ('line\n' * 201, 'x' * 65537):
            assert 'exceeds' in str(
                await tool.invoke(params(content).model_dump_json()),
            )
    assert memory.docs['existing-user'][0] == 'original brief'


@pytest.mark.parametrize(
    ('name', 'payload'),
    [
        ('update_agents_file', {'content': 'bypass'}),
        ('write_memory_note', {'path': 'Agents/AGENTS.md', 'content': 'bypass'}),
        ('edit_memory_note', {'path': 'Agents/AGENTS.md'}),
        ('delete_memory_notes', {'paths': ['Agents/AGENTS.md']}),
        ('rename_memory_note', {'from': 'Agents/note.md', 'to': 'Agents/AGENTS.md'}),
        ('rename_memory_note', {'from': 'Agents', 'to': 'elsewhere'}),
    ],
)
async def test_raw_memory_tools_cannot_bypass_managed_brief(
    name: str,
    payload: dict[str, JsonValue],
) -> None:
    memory = VersionedMemory()
    tool = MCPTool(
        memory,
        'memory',
        MCPToolInfo(name=name, description='test', input_schema={}),
        {'user': 'existing-user'},
    )
    await tool.invoke(json.dumps(payload))
    assert not memory.calls


async def test_ordinary_memory_notes_can_reference_the_brief() -> None:
    memory = VersionedMemory()
    tool = MCPTool(
        memory,
        'memory',
        MCPToolInfo(name='write_memory_note', description='test', input_schema={}),
        {'user': 'existing-user'},
    )
    await tool.invoke(
        json.dumps(
            {'path': 'Agents/projects/topic.md', 'content': 'See Agents/AGENTS.md'},
        ),
    )
    assert len(memory.calls) == 1


async def test_brief_changes_leave_the_static_cache_prefix_identical() -> None:
    from app.features.agents.entities import TurnInput
    from app.features.agents.services import AgentRunner
    from app.features.agents.tooling import ToolSet
    from lib.llm import ModelCapabilities, TextChunk
    from tests.helpers.replies import RecordingReply
    from tests.unit.app.features.agents.test_agent_runner import FakeLLM

    llm = FakeLLM([[TextChunk(text='first')], [TextChunk(text='second')]])
    runner = AgentRunner(
        llm,
        ToolSet(logger=make_logger()),
        FakeClock(),
        logger=make_logger(),
    )
    turn = TurnInput(
        instructions='unchanging static policy',
        dynamic_instructions=('brief version one',),
        prompts=(TurnPrompt(text='hello', actor='alice'),),
        model='parent',
        capabilities=ModelCapabilities(),
        now=FakeClock().now(),
        channel='test',
    )
    await runner.run(turn, RecordingReply())
    await runner.run(
        turn.model_copy(update={'dynamic_instructions': ('brief version two',)}),
        RecordingReply(),
    )
    assert llm.requests[0].messages[0] == llm.requests[1].messages[0]
    assert llm.requests[0].messages[0].cacheable
    assert not llm.requests[0].messages[1].cacheable
    assert llm.requests[0].messages[1] != llm.requests[1].messages[1]


@pytest.mark.parametrize(
    ('text', 'quote'),
    [
        ('I do not want you to update AGENTS.md', None),
        ("I don't want you to update AGENTS.md", None),
        ('I don\u2019t want you to update AGENTS.md', None),
        ('There is no need to update AGENTS.md', None),
        ("Please don't ever let anyone update AGENTS.md", None),
        ('No one should update AGENTS.md', None),
        ('我不希望你在今天的這次對話中更新 AGENTS.md', None),
        ('不要更新 AGENTS.md，記錄在別的筆記就好', None),  # noqa: RUF001
        ("Please don't update AGENTS.md, record it elsewhere", None),
        ('AGENTS.md 先不動，我記錄在 HEARTBEAT', None),  # noqa: RUF001
        ('Avoid updating AGENTS.md; add it to notes', None),
        ('暫緩更新 AGENTS.md', None),
        ('等等再更新 AGENTS.md', None),
        ('無需更新 AGENTS.md', None),
        ('我昨天已經更新 AGENTS.md 了', None),
        ('He said: update AGENTS.md', None),
        ('Update AGENTS.md, actually never mind.', None),
        # A quote cannot span sentences or drop the negation that governs it.
        (
            '先不要更新 AGENTS.md。剛剛的討論我有記錄下來了。',
            'AGENTS.md。剛剛的討論我有記錄',
        ),
        (
            'Never update AGENTS.md. We record decisions in notes.',
            'AGENTS.md. We record',
        ),
        ('先不更新 AGENTS.md，只討論內容。', '更新 AGENTS.md'),  # noqa: RUF001
        ('請不要更新 AGENTS.md', '更新 AGENTS.md'),
    ],
)
def test_negated_reported_and_conditional_sentences_do_not_authorize(
    text: str,
    quote: str | None,
) -> None:
    from app.features.briefs.tools import _requested

    prompt = TurnPrompt(text=text, actor='alice', message_id='msg-1')
    assert not _requested(prompt, quote or text)


@pytest.mark.parametrize(
    ('text', 'quote'),
    [
        ('請更新 AGENTS.md：不要自動部署。', None),  # noqa: RUF001
        ('記住：不要再用 tabs。請把這條加到 AGENTS.md', '請把這條加到 AGENTS.md'),  # noqa: RUF001
        ('請把這個決定寫進協作簡報', None),
        ('不錯，請更新 AGENTS.md', None),  # noqa: RUF001
        ('可以幫我更新 AGENTS.md 嗎？', None),  # noqa: RUF001
        ('Can you update AGENTS.md with the decision?', None),
        ('更新 AGENTS.md，記錄如何部署', None),  # noqa: RUF001
        ('更新 AGENTS.md，以後不要再自動部署', None),  # noqa: RUF001
        ('把「不要用簡體」記錄到 AGENTS.md', None),
        ('在 AGENTS.md 特別記錄部署流程', None),
        ('把我說的部署流程寫到 AGENTS.md', None),
        ('Change AGENTS.md to say we deploy on Fridays.', None),
    ],
)
def test_explicit_and_polite_requests_authorize(text: str, quote: str | None) -> None:
    from app.features.briefs.tools import _requested

    prompt = TurnPrompt(text=text, actor='alice', message_id='msg-1')
    assert _requested(prompt, quote or text)
