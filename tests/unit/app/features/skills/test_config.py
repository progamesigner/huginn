import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config.exceptions import ConfigError, MissingEnvironmentVariableError
from app.core.config.loader import load_config
from app.core.config.registry import HuginnConfig, SectionRegistry
from app.features.agents.config import AGENTS
from app.features.agents.config import SkillsConfig as AgentSkills
from app.features.mcp.config import MCP_SERVERS
from app.features.mcp.exceptions import UnknownMCPServerReferenceError
from app.features.skills.config import SKILLS, MCPSkillSourceConfig, SkillsConfig
from app.features.skills.exceptions import SkillsNotConfiguredError

_SOURCE = {'kind': 'mcp', 'server': 'memory'}


def load(
    tmp_path: Path,
    skills: Mapping[str, object],
    agent_skills: Mapping[str, object],
    environment: dict[str, str] | None = None,
) -> HuginnConfig:
    path = tmp_path / 'config.yaml'
    path.write_text(
        json.dumps(
            {
                'agents': {
                    'scout': {
                        'description': 'Research',
                        'instructions': 'Research',
                        'skills': agent_skills,
                    },
                },
                'mcp_servers': {
                    'memory': {
                        'transport': 'http',
                        'url': 'https://memory.test',
                    },
                },
                'skills': skills,
            },
        ),
    )
    return load_config(
        path,
        SectionRegistry((AGENTS, MCP_SERVERS, SKILLS)),
        environment={} if environment is None else environment,
    )


@pytest.mark.parametrize('legacy', [True, False])
def test_default_source_is_selected_without_an_agent_reference(
    tmp_path: Path,
    legacy: bool,
) -> None:
    skills = {'source': _SOURCE} if legacy else {'sources': {'default': _SOURCE}}
    config = load(tmp_path, skills, {'enabled': True})
    source = config.section(SKILLS).resolve_source('default')
    assert isinstance(source, MCPSkillSourceConfig)
    assert source.server == 'memory'


def test_named_source_needs_no_default(tmp_path: Path) -> None:
    config = load(
        tmp_path,
        {'sources': {'advisor': _SOURCE}},
        {
            'enabled': True,
            'source': 'advisor',
        },
    )
    assert config.section(AGENTS).root['scout'].skills.source == 'advisor'
    assert set(config.section(SKILLS).named_sources) == {'advisor'}


def test_legacy_default_can_coexist_with_other_named_sources(tmp_path: Path) -> None:
    config = load(
        tmp_path,
        {
            'source': _SOURCE,
            'sources': {'advisor': _SOURCE},
        },
        {'enabled': True, 'source': 'advisor'},
    )
    assert set(config.section(SKILLS).named_sources) == {'default', 'advisor'}


@pytest.mark.parametrize('enabled', [True, False])
def test_unknown_source_reference_fails_even_when_disabled(
    tmp_path: Path,
    enabled: bool,
) -> None:
    with pytest.raises(ConfigError, match=r'agents\.scout\.skills\.source.*missing'):
        load(
            tmp_path,
            {'sources': {'default': _SOURCE}},
            {
                'enabled': enabled,
                'source': 'missing',
            },
        )


@pytest.mark.parametrize('skills', [{}, {'sources': {'advisor': _SOURCE}}])
def test_enabled_skills_require_the_selected_default(
    tmp_path: Path,
    skills: Mapping[str, object],
) -> None:
    with pytest.raises(SkillsNotConfiguredError, match=r'agents\.scout.*default'):
        load(tmp_path, skills, {'enabled': True})


def test_disabled_skills_need_no_source(tmp_path: Path) -> None:
    load(tmp_path, {}, {})


def test_ambiguous_default_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match='cannot both be set'):
        load(tmp_path, {'source': _SOURCE, 'sources': {'default': _SOURCE}}, {})


def test_all_named_sources_validate_mcp_references(tmp_path: Path) -> None:
    with pytest.raises(
        UnknownMCPServerReferenceError,
        match=r'skills.sources.unused.server',
    ):
        load(
            tmp_path,
            {
                'sources': {
                    'unused': {'kind': 'mcp', 'server': 'missing'},
                },
            },
            {},
        )


def test_named_sources_use_hook_value_resolution(tmp_path: Path) -> None:
    scope_file = tmp_path / 'user'
    scope_file.write_text('alice\n')
    config = load(
        tmp_path,
        {
            'sources': {
                'advisor': {
                    **_SOURCE,
                    'glob': '${SKILL_ROOT}/**/SKILL.md',
                    'inject': {
                        'agent': '${SCOPE_AGENT}',
                        'user': {'value_from': {'file': str(scope_file)}},
                        'context': {'roles': ['${SCOPE_AGENT}'], 'limit': 10},
                    },
                },
            },
        },
        {'enabled': True, 'source': 'advisor'},
        {
            'SKILL_ROOT': 'Agents/skills',
            'SCOPE_AGENT': 'advisor',
        },
    )
    source = config.section(SKILLS).resolve_source('advisor')
    assert isinstance(source, MCPSkillSourceConfig)
    assert source.glob == 'Agents/skills/**/SKILL.md'
    assert source.inject == {
        'agent': 'advisor',
        'user': 'alice',
        'context': {'roles': ['advisor'], 'limit': 10},
    }


def test_missing_source_environment_variable_names_the_field(tmp_path: Path) -> None:
    with pytest.raises(
        MissingEnvironmentVariableError,
        match=r'skills.sources.advisor.inject.agent',
    ):
        load(
            tmp_path,
            {
                'sources': {
                    'advisor': {
                        **_SOURCE,
                        'inject': {'agent': '${MISSING}'},
                    },
                },
            },
            {},
        )


def test_named_source_schema_and_injection_validation() -> None:
    schema = SkillsConfig.model_json_schema()
    assert 'sources' in schema['properties']
    source_schema = schema['properties']['sources']['additionalProperties']
    assert source_schema == {'$ref': '#/$defs/SkillSourceConfig'}
    assert (
        schema['$defs']['SkillSourceConfig']['discriminator']['propertyName'] == 'kind'
    )
    assert 'source' in AgentSkills.model_json_schema()['properties']
    with pytest.raises(ValidationError, match='inject and bind overlap'):
        SkillsConfig.model_validate_json(
            json.dumps(
                {
                    'sources': {
                        'advisor': {
                            **_SOURCE,
                            'inject': {'agent': 'advisor'},
                            'bind': {'agent': 'agent'},
                        },
                    },
                },
            ),
        )
    with pytest.raises(ValidationError, match='must not be empty'):
        SkillsConfig.model_validate_json(json.dumps({'sources': {'': _SOURCE}}))
    with pytest.raises(ValidationError, match='at least 1 character'):
        AgentSkills(source='')


def test_named_filesystem_source_resolves_root_and_glob_values(tmp_path: Path) -> None:
    config = load(
        tmp_path,
        {
            'sources': {
                'local': {
                    'kind': 'filesystem',
                    'root': '${SKILL_ROOT}',
                    'glob': {'value_from': {'env': 'SKILL_GLOB'}},
                },
            },
        },
        {'enabled': True, 'source': 'local'},
        {
            'SKILL_ROOT': str(tmp_path),
            'SKILL_GLOB': '**/SKILL.md',
        },
    )
    from app.features.skills.config import FilesystemSkillSourceConfig

    source = config.section(SKILLS).resolve_source('local')
    assert isinstance(source, FilesystemSkillSourceConfig)
    assert source.root == tmp_path
    assert source.glob == '**/SKILL.md'


@pytest.mark.parametrize('legacy_shape', ['source', 'sources'])
async def test_migrated_asset_sources_preserve_effective_role_bindings(
    tmp_path: Path,
    legacy_shape: str,
) -> None:
    from app.features.assets.config import ASSETS
    from app.features.skills.config import effective_skills

    original = {
        'kind': 'mcp',
        'server': 'memory',
        'glob': 'Agents/custom/**/SKILL.md',
        'inject': {'agent': 'advisor'},
        'bind': {'user': 'handle'},
    }
    payload = {
        'agents': {
            'advisor': {
                'description': 'Advice',
                'instructions': 'Help',
                'skills': {'enabled': True, 'source': 'default', 'allow': ['review']},
            },
        },
        'mcp_servers': {'memory': {'transport': 'http', 'url': 'https://memory.test'}},
        'skills': {
            legacy_shape: original
            if legacy_shape == 'source'
            else {'default': original},
        },
    }
    path = tmp_path / 'config.yaml'
    path.write_text(json.dumps(payload))
    registry = SectionRegistry((AGENTS, MCP_SERVERS, SKILLS, ASSETS))
    before = load_config(path, registry)
    migrated = {key: value for key, value in original.items() if key != 'glob'}
    migrated['globs'] = {'skill': original['glob']}
    payload['skills'] = {}
    payload['assets'] = {'sources': {'default': migrated}}
    path.write_text(json.dumps(payload))
    after = load_config(path, registry)
    assert (
        effective_skills(before).named_sources == effective_skills(after).named_sources
    )
    assert (
        before.section(AGENTS).root['advisor'].skills
        == after.section(AGENTS).root['advisor'].skills
    )
