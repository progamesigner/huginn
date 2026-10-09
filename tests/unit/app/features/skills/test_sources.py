import json
from collections.abc import Mapping
from pathlib import Path
from typing import override

import pytest
from pydantic import JsonValue, ValidationError

from app.core.config.loader import load_config
from app.core.config.registry import SectionRegistry
from app.features.agents.config import AGENTS, AgentConfig
from app.features.agents.config import SkillsConfig as AgentSkills
from app.features.agents.entities import TurnScope
from app.features.skills.config import SKILLS, FilesystemSkillSourceConfig, SkillsConfig
from app.features.skills.repositories import SkillRepository
from app.features.skills.sources.filesystem import FilesystemSkillSource
from app.features.skills.sources.mcp import MCPSkillSource
from app.features.skills.tools import SkillToolSource
from app.providers.contributions import (
    AgentContributions,
    CoreContributions,
    SkillContributions,
)
from tests.helpers.logging import make_logger
from tests.helpers.settings import app_with, make_settings
from tests.unit.app.features.skills.test_mcp_source import FakeMCP


class CustomMCP(FakeMCP):
    @override
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        self.calls.append((server, tool, dict(arguments)))
        assert server == 'other'
        if tool == 'catalog':
            return json.dumps({'items': ['skills/triage/SKILL.md']})
        assert tool == 'open_note'
        return json.dumps(
            {'content': '---\nname: triage\ndescription: Triage issues\n---\nSteps'},
        )


async def test_configurable_mcp_tools_and_server() -> None:
    client = CustomMCP()
    source = MCPSkillSource(
        client,
        'other',
        '**/SKILL.md',
        list_tool='catalog',
        read_tool='open_note',
        logger=make_logger(),
    )
    inventory = await source.list({'user': 'alice'})
    loaded = await source.load(inventory[0].path, {'user': 'alice'})
    assert loaded.body == 'Steps'
    assert [call[1] for call in client.calls] == ['catalog', 'open_note', 'open_note']
    assert all(call[2]['user'] == 'alice' for call in client.calls)


def write_skill(
    root: Path,
    name: str = 'triage',
    content: str = '---\nname: triage\ndescription: Triage issues\n---\nSteps',
) -> Path:
    path = root / name / 'SKILL.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


async def test_filesystem_inventory_skips_bad_frontmatter_and_loads_body(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    write_skill(tmp_path)
    write_skill(tmp_path, 'bad', '---\nname: [invalid\n---\nbad')
    write_skill(tmp_path, 'scalar', '---\nscalar\n---\nbad')
    write_skill(tmp_path, 'unterminated', '---\nname: bad')
    source = FilesystemSkillSource(tmp_path, '**/SKILL.md', logger=make_logger())
    with caplog.at_level('WARNING'):
        inventory = await source.list({})
    assert [m.name for m in inventory] == ['triage']
    assert len(caplog.records) == 3
    skill = await source.load(inventory[0].path, {})
    assert skill.body == 'Steps'
    assert skill.meta.description == 'Triage issues'


async def test_filesystem_rejects_load_traversal_and_symlinks(tmp_path: Path) -> None:
    root = tmp_path / 'skills'
    root.mkdir()
    outside = write_skill(tmp_path / 'outside')
    source = FilesystemSkillSource(root, '**/SKILL.md', logger=make_logger())
    with pytest.raises(ValueError, match='escapes'):
        await source.load('../outside/triage/SKILL.md', {})
    (root / 'SKILL.md').symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        await source.list({})
    with pytest.raises(ValueError, match='escapes'):
        await source.load('SKILL.md', {})


@pytest.mark.parametrize('glob', ['../**/SKILL.md', '/etc/*'])
def test_filesystem_rejects_glob_escape(tmp_path: Path, glob: str) -> None:
    with pytest.raises(ValueError, match='within its root'):
        FilesystemSkillSource(tmp_path, glob, logger=make_logger())
    with pytest.raises(ValidationError, match='within its root'):
        FilesystemSkillSourceConfig(kind='filesystem', root=tmp_path, glob=glob)


@pytest.mark.parametrize(
    'extra',
    [{'inject': {'user': 'alice'}}, {'bind': {'user': 'handle'}}],
)
def test_filesystem_config_rejects_scope_fields(
    tmp_path: Path,
    extra: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match='Extra inputs'):
        SkillsConfig.model_validate_json(
            json.dumps(
                {'source': {'kind': 'filesystem', 'root': str(tmp_path), **extra}},
            ),
        )


def test_legacy_flat_config_points_to_source_migration() -> None:
    with pytest.raises(ValidationError, match=r'skills\.source'):
        SkillsConfig.model_validate_json('{"server":"memory"}')


async def test_filesystem_tool_enabled_allow_and_cached_inventory_without_mcp(
    tmp_path: Path,
) -> None:
    write_skill(tmp_path)
    path = tmp_path / 'config.yaml'
    path.write_text(
        f'agents: {{scout: {{description: x, instructions: y, model: test, skills: {{enabled: true}}}}}}\nskills:\n  source:\n    kind: filesystem\n    root: {tmp_path}\n',
    )
    # No mcp_servers section is needed even when skills are enabled.
    load_config(path, SectionRegistry((AGENTS, SKILLS)))
    app = app_with(make_settings(config_file=path)).build(
        AgentContributions(),
        CoreContributions(),
        SkillContributions(),
    )
    repository = app.resolve(SkillRepository)
    tools = SkillToolSource(
        lambda: repository,
        SkillsConfig(
            source=FilesystemSkillSourceConfig(kind='filesystem', root=tmp_path),
        ),
        logger=make_logger(),
    )
    scope = TurnScope(agent='scout', actor='user:1', channel='test', chat='1')
    agent = AgentConfig(
        description='x',
        instructions='y',
        skills=AgentSkills(enabled=True, allow=('triage',)),
    )
    harvest = await tools.collect(agent, scope)
    assert await harvest.tools[0][1].invoke('{"name":"triage"}') == 'Steps'
    assert 'unknown skill' in await harvest.tools[0][1].invoke('{"name":"other"}')
    write_skill(tmp_path, 'later', '---\nname: later\ndescription: Later\n---\nNew')
    assert len(await repository.inventory({})) == 1
    disabled = agent.model_copy(update={'skills': AgentSkills(enabled=False)})
    assert (await tools.collect(disabled, scope)).tools == ()


async def test_local_read_failure_is_an_error(tmp_path: Path) -> None:
    source = FilesystemSkillSource(tmp_path, '**/SKILL.md', logger=make_logger())
    with pytest.raises(FileNotFoundError):
        await source.load('missing.md', {})
    with pytest.raises(FileNotFoundError):
        FilesystemSkillSource(tmp_path / 'missing', '**/SKILL.md', logger=make_logger())


async def test_local_inventory_failure_is_not_reported_as_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    from app.features.skills.repositories import SkillRepository
    from tests.helpers.clock import FakeClock

    source = FilesystemSkillSource(tmp_path, '**/SKILL.md', logger=make_logger())
    monkeypatch.setattr(
        source,
        'list',
        AsyncMock(side_effect=OSError('local read failure')),
    )
    repository = SkillRepository(source, FakeClock())
    tools = SkillToolSource(
        lambda: repository,
        SkillsConfig(
            source=FilesystemSkillSourceConfig(kind='filesystem', root=tmp_path),
        ),
        logger=make_logger(),
    )
    agent = AgentConfig(
        description='x',
        instructions='y',
        skills=AgentSkills(enabled=True),
    )
    with pytest.raises(OSError, match='local read failure'):
        await tools.collect(
            agent,
            TurnScope(agent='scout', actor='user:1', channel='test', chat='1'),
        )


async def test_frontmatter_delimiters_are_lines_and_empty_frontmatter_is_valid(
    tmp_path: Path,
) -> None:
    write_skill(
        tmp_path,
        content='---\nname: triage\ndescription: "before---after"\n---\nSteps',
    )
    write_skill(tmp_path, 'empty', '---\n---\nPlain steps')
    source = FilesystemSkillSource(tmp_path, '**/SKILL.md', logger=make_logger())
    inventory = await source.list({})
    assert {meta.name: meta.description for meta in inventory} == {
        'empty': 'empty',
        'triage': 'before---after',
    }
