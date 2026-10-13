from pathlib import Path

import pytest

from app.core.application import App
from app.features.agents.contracts import ToolSourceRegistry
from app.features.skills.exceptions import SkillsNotConfiguredError
from app.features.skills.repositories import SkillRepositories, SkillRepository
from app.features.skills.tools import SkillToolSource
from app.providers.contributions import (
    AgentContributions,
    CoreContributions,
    MCPContributions,
    SkillContributions,
)
from tests.helpers.settings import app_with, make_settings

_CONFIGURED = """
defaults:
  model: gpt-5-mini

agents:
  scout:
    description: Research agent
    instructions: You are scout.

mcp_servers:
  memory:
    transport: http
    url: https://memory.test

skills:
  source:
    kind: mcp
    server: memory
    inject:
      agent: default
"""

_UNCONFIGURED = """
defaults:
  model: gpt-5-mini

agents:
  scout:
    description: Research agent
    instructions: You are scout.
"""


def build(path: Path) -> App:
    return app_with(make_settings(config_file=path)).build(
        AgentContributions(),
        CoreContributions(),
        MCPContributions(),
        SkillContributions(),
    )


def test_repository_resolves_when_a_source_is_configured(tmp_path: Path) -> None:
    config = tmp_path / 'huginn.yaml'
    config.write_text(_CONFIGURED)

    container = build(config)

    assert isinstance(
        container.resolve(SkillRepositories).get('default'),
        SkillRepository,
    )


def test_repository_is_shared(tmp_path: Path) -> None:
    config = tmp_path / 'huginn.yaml'
    config.write_text(_CONFIGURED)

    container = build(config)

    # NOTE: the TTL cache is worthless unless every caller gets the same instance
    assert container.resolve(SkillRepositories).get('default') is container.resolve(
        SkillRepositories,
    ).get('default')


def test_repository_reports_an_unconfigured_source_on_resolve(tmp_path: Path) -> None:
    config = tmp_path / 'huginn.yaml'
    config.write_text(_UNCONFIGURED)

    container = build(config)

    with pytest.raises(SkillsNotConfiguredError):
        container.resolve(SkillRepositories).get('default')


def test_provider_binds_without_a_config_file(tmp_path: Path) -> None:
    container = build(tmp_path / 'does-not-exist.yaml')

    assert container.bound(SkillRepositories)


def test_provider_contributes_a_tool_source(tmp_path: Path) -> None:
    config = tmp_path / 'huginn.yaml'
    config.write_text(_CONFIGURED)

    container = build(config)

    sources = container.resolve(ToolSourceRegistry).sources()

    assert any(isinstance(source, SkillToolSource) for source in sources)


@pytest.mark.parametrize('mode', ['scopes', 'servers', 'shared'])
async def test_agents_keep_inventory_load_and_cache_in_their_selected_source(
    tmp_path: Path,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from collections.abc import Mapping
    from typing import override

    from pydantic import JsonValue

    from app.core.config.registry import HuginnConfig
    from app.features.agents.config import AGENTS
    from app.features.agents.entities import TurnScope
    from app.features.skills.config import SKILLS
    from lib.mcp import MCPClientContract
    from tests.helpers.logging import make_logger
    from tests.unit.app.features.skills.test_mcp_source import FakeMCP

    class ScopedMCP(FakeMCP):
        @override
        async def call_tool(
            self,
            server: str,
            tool: str,
            arguments: Mapping[str, JsonValue],
        ) -> str:
            self.calls.append((server, tool, dict(arguments)))
            if tool == 'list_memory_notes':
                return json.dumps({'items': ['Agents/skills/shared/SKILL.md']})
            marker = f'{server}/{arguments["agent"]}/{arguments["user"]}'
            return json.dumps(
                {
                    'content': f'---\nname: shared\ndescription: {marker}\n---\n{marker}',
                },
            )

    monkeypatch.setenv('TEST_ADVISOR_SCOPE', 'advisor')
    monkeypatch.setenv('TEST_REVIEWER_SCOPE', 'reviewer')
    sources = {
        name: {
            'kind': 'mcp',
            'server': name if mode == 'servers' else 'memory',
            'glob': f'Agents/{name}/**/SKILL.md',
            'inject': {
                'agent': '${TEST_' + name.upper() + '_SCOPE}'
                if mode == 'scopes'
                else 'shared',
                'user': 'alice',
            },
        }
        for name in ('advisor', 'reviewer')
    }
    if mode == 'shared':
        sources = {
            'shared': {
                'kind': 'mcp',
                'server': 'memory',
                'inject': {'user': 'alice'},
                'bind': {'agent': 'agent'},
            },
        }
    path = tmp_path / 'config.yaml'
    path.write_text(
        json.dumps(
            {
                'defaults': {'model': 'test'},
                'agents': {
                    name: {
                        'description': name,
                        'instructions': 'Help',
                        'skills': {
                            'enabled': True,
                            'source': 'shared' if mode == 'shared' else name,
                        },
                    }
                    for name in ('advisor', 'reviewer')
                },
                'mcp_servers': {
                    name: {'transport': 'http', 'url': f'https://{name}.test'}
                    for name in ('memory', 'advisor', 'reviewer')
                },
                'skills': {'sources': sources},
            },
        ),
    )
    container = build(path)
    client = ScopedMCP()
    with container.use(MCPClientContract, client):
        config = container.resolve(HuginnConfig)
        repositories = container.resolve(SkillRepositories)
        assert repositories is container.resolve(SkillRepositories)
        tools = SkillToolSource(
            repositories.get,
            config.section(SKILLS),
            logger=make_logger(),
        )
        for name in ('advisor', 'reviewer', 'advisor'):
            scope = TurnScope(agent=name, actor='user:1', channel='test', chat='1')
            harvest = await tools.collect(config.section(AGENTS).root[name], scope)
            expected = (
                f'{name}/shared/alice' if mode == 'servers' else f'memory/{name}/alice'
            )
            assert expected in harvest.tools[0][1].description
            assert await harvest.tools[0][1].invoke('{"name":"shared"}') == expected
        lists = [call for call in client.calls if call[1] == 'list_memory_notes']
        assert len(lists) == 2
        reads = [call for call in client.calls if call[1] == 'read_memory_note']
        assert len(reads) == 5
        for call in reads:
            assert (call[0], call[2]['agent'], call[2]['user']) in {
                (listed[0], listed[2]['agent'], listed[2]['user']) for listed in lists
            }
        if mode != 'shared':
            assert {call[2]['glob'] for call in lists} == {
                'Agents/advisor/**/SKILL.md',
                'Agents/reviewer/**/SKILL.md',
            }
            assert repositories.get('advisor') is not repositories.get('reviewer')
