import json
from collections.abc import Mapping
from typing import override

import pytest
from pydantic import JsonValue

from app.core.config.registry import HuginnConfig
from app.features.agents.config import AGENTS, AgentConfigList
from app.features.agents.entities import (
    RoutingPolicy,
    SelectionPolicy,
    TurnPrompt,
    TurnRequest,
)
from app.features.assets.config import AssetsConfig, MCPAssetSourceConfig
from app.features.assets.services import AssetsService
from app.features.channels.scoped_commands import ScopedCommands
from tests.helpers.clock import FakeClock
from tests.helpers.logging import make_logger
from tests.unit.app.features.assets.test_assets_service import Memory


class Templates(Memory):
    def __init__(self) -> None:
        super().__init__()
        self.target: str | None = None
        self.command = 'review'
        self.template = 'Review $ARGUMENTS'

    @override
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        self.calls.append((tool, dict(arguments)))
        if tool == 'list_memory_notes':
            return json.dumps({'items': [f'Agents/commands/{self.command}.md']})
        metadata = '---\ndescription: Review work\n'
        if self.target is not None:
            metadata += f'agent: {self.target}\n'
        return json.dumps(
            {'content': metadata + f'---\n{arguments.get("user")}: ' + self.template},
        )


def request(
    user: str | None = 'old-memory',
    text: str = '/review raw $ARGUMENTS',
) -> TurnRequest:
    return TurnRequest(
        channel='bot',
        chat='42',
        policy=RoutingPolicy(agent='writer', allowed=frozenset({'writer', 'reviewer'})),
        prompts=(
            TurnPrompt(
                text=text,
                actor='alice',
                account='1' if user else None,
                handle='new-handle' if user else None,
                memory_scope=user,
            ),
        ),
    )


def resolver(memory: Templates, *, alternate: bool = False) -> ScopedCommands:
    settings = AssetsConfig(
        sources={
            'one': MCPAssetSourceConfig(
                server='memory',
                inject={'agent': 'default'},
                bind={'user': 'memory_scope'},
            ),
            'two': MCPAssetSourceConfig(
                server='memory',
                inject={'agent': 'reviewer'},
                bind={'user': 'memory_scope'},
            ),
        },
    )
    agents = AgentConfigList.model_validate_json(
        json.dumps(
            {
                name: {
                    'description': name,
                    'instructions': name,
                    'commands': {
                        'enabled': True,
                        'source': 'two' if name == 'reviewer' and alternate else 'one',
                    },
                }
                for name in ('writer', 'reviewer')
            },
        ),
    )
    return ScopedCommands(
        HuginnConfig({AGENTS.key: agents}),
        settings,
        AssetsService(settings, memory, FakeClock(), logger=make_logger()),
        frozenset({'help', 'new'}),
        logger=make_logger(),
    )


async def test_same_command_uses_personal_binding_for_inventory_help_and_fresh_body() -> (
    None
):
    memory = Templates()
    commands = resolver(memory)
    for user in ('old-memory', 'bob'):
        assert [entry.name for entry in await commands.inventory(request(user))] == [
            'review',
        ]
        expanded = await commands.resolve(request(user))
        assert expanded.prompts[0].text == f'{user}: Review raw $ARGUMENTS'
        assert expanded.prompts[0].original_text == '/review raw $ARGUMENTS'
        assert expanded.prompts[0].origin == 'template'
    memory.template = 'Edited without placeholder'
    assert (await commands.resolve(request())).prompts[
        0
    ].text == 'old-memory: Edited without placeholder\n\nraw $ARGUMENTS'
    assert not await commands.inventory(request(None))
    assert all(call[1].get('user') in {'old-memory', 'bob'} for call in memory.calls)


async def test_disabled_and_revoked_permissions_take_effect_without_expiring_inventory() -> (
    None
):
    commands = resolver(Templates())
    await commands.inventory(request())
    agents = commands.config.section(AGENTS).root
    for name, agent in list(agents.items()):
        agents[name] = agent.model_copy(
            update={
                'commands': agent.commands.model_copy(update={'allow': ('other',)}),
            },
        )
    assert not await commands.inventory(request())
    # A revoked command is no longer expanded; the text stays ordinary input.
    assert await commands.resolve(request()) == request()


async def test_slash_text_that_names_no_command_stays_ordinary_input() -> None:
    commands = resolver(Templates())
    plain = request(text='/summarize this article')
    assert await commands.resolve(plain) == plain
    agents = commands.config.section(AGENTS).root
    for name, agent in list(agents.items()):
        agents[name] = agent.model_copy(
            update={'commands': agent.commands.model_copy(update={'enabled': False})},
        )
    assert not commands.enabled(request().policy)
    assert await commands.resolve(request()) == request()


async def test_target_agent_is_checked_again_from_fresh_metadata() -> None:
    memory = Templates()
    commands = resolver(memory)
    await commands.inventory(request())
    memory.target = 'outsider'
    with pytest.raises(ValueError, match='unauthorized'):
        await commands.resolve(request())
    memory.target = 'reviewer'
    result = await commands.resolve(request())
    assert result.policy.agent == 'reviewer'
    assert result.policy.allowed == frozenset({'reviewer'})


async def test_conflicting_and_reserved_names_are_withheld_alone() -> None:
    assert not await resolver(Templates(), alternate=True).inventory(request())
    memory = Templates()
    memory.command = 'help'
    commands = resolver(memory)
    assert not await commands.inventory(request())
    # The reserved name stays the built-in; the template is never expanded.
    assert await commands.resolve(request(text='/help')) == request(text='/help')


async def test_router_is_narrowed_without_switching_template_source() -> None:
    commands = resolver(Templates())
    routed = request().model_copy(
        update={
            'policy': RoutingPolicy(
                selection=SelectionPolicy(
                    fallback='writer',
                    allowed=frozenset({'writer', 'reviewer'}),
                ),
                allowed=frozenset({'writer', 'reviewer'}),
            ),
        },
    )
    result = await commands.resolve(routed)
    assert result.policy.agent is None
    assert result.policy.allowed == frozenset({'writer', 'reviewer'})
    assert result.policy.selection is not None
    assert result.policy.selection.allowed == result.policy.allowed


async def test_mixed_batch_does_not_lend_personal_scope_to_another_sender() -> None:
    commands = resolver(Templates())
    mixed = request().model_copy(
        update={'prompts': (*request().prompts, TurnPrompt(text='hello', actor='bob'))},
    )
    assert not await commands.inventory(mixed)
    with pytest.raises(ValueError, match='mixes'):
        await commands.resolve(mixed)


async def test_http_slash_text_reaches_the_model_when_no_command_matches() -> None:
    from lib.llm import TextChunk
    from tests.helpers.replies import RecordingReply
    from tests.unit.app.features.agents.test_agent_runner import FakeLLM
    from tests.unit.app.features.agents.test_turn_handler import (
        FakeStore,
        configured_handler,
    )
    from tests.unit.app.features.agents.test_turn_handler import (
        request as http_request,
    )

    llm = FakeLLM([[TextChunk(text='summary')]])
    handler = configured_handler(llm, FakeStore())
    handler.commands = resolver(Templates())
    reply = RecordingReply()
    await handler.handle(http_request('/summarize this article'), reply)
    assert 'summary' in ''.join(reply.fragments)
    assert len(llm.requests) == 1


async def test_mixed_batch_with_unknown_slash_text_is_not_rejected() -> None:
    commands = resolver(Templates())
    mixed = request(text='/summarize this').model_copy(
        update={
            'prompts': (
                *request(text='/summarize this').prompts,
                TurnPrompt(text='hello', actor='bob'),
            ),
        },
    )
    assert await commands.resolve(mixed) == mixed


async def test_an_unavailable_inventory_is_answered_rather_than_failing_the_turn() -> (
    None
):
    class Down(Templates):
        @override
        async def call_tool(
            self,
            server: str,
            tool: str,
            arguments: Mapping[str, JsonValue],
        ) -> str:
            raise RuntimeError('memory service down')

    with pytest.raises(ValueError, match='temporarily unavailable'):
        await resolver(Down()).resolve(request())
