from pathlib import Path
from typing import Annotated, override

import pytest
from pydantic import Field, HttpUrl

from app.core.config import HuginnConfig, SectionRegistry
from app.core.config.exceptions import (
    InvalidValueReferenceError,
    MissingConfigFileError,
    MissingEnvironmentVariableError,
    MissingSectionError,
    UnknownSectionError,
)
from app.core.config.loader import (
    interpolate,
    load_config,
    resolve_references,
    resolve_value_from,
)
from app.features.agents.config import (
    AgentConfig,
    AgentsConfig,
    ChannelConfig,
    ChannelRouterConfig,
    ChannelsConfig,
    DefaultsConfig,
    McpRef,
    McpServerConfig,
    McpServersConfig,
    ToolRef,
    build_sections,
    validate_agents,
    validate_channels,
)
from app.features.agents.exceptions import (
    AmbiguousInstructionsError,
    UnknownAgentReferenceError,
    UnknownMcpServerReferenceError,
    UnknownToolReferenceError,
)
from app.features.agents.tools import Tool, ToolOutput, ToolParams, ToolRegistry


class SearchParams(ToolParams):
    query: Annotated[str, Field()]


class WebSearchTool(Tool[SearchParams]):
    name = 'websearch'
    description = 'Fake tool for config loader tests'
    params = SearchParams

    @override
    async def run(self, params: SearchParams) -> ToolOutput:
        return 'ok'


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool)
    return registry


def build_section_registry() -> SectionRegistry:
    registry = SectionRegistry()
    for spec in build_sections(build_tool_registry()):
        registry.register(spec)
    return registry


def build_agent(**overrides: object) -> AgentConfig:
    fields: dict[str, object] = {
        'description': 'General-purpose research agent',
        'instructions': 'You are scout.',
    }
    fields.update(overrides)
    return AgentConfig.model_validate(fields)


def build_app_config(
    agents: dict[str, AgentConfig] | None = None,
    mcp_servers: dict[str, McpServerConfig] | None = None,
    channels: ChannelsConfig | None = None,
) -> HuginnConfig:
    return HuginnConfig(
        {
            'agents': AgentsConfig(agents or {}),
            'mcp_servers': McpServersConfig(mcp_servers or {}),
            'channels': channels or ChannelsConfig(),
        }
    )


# interpolate()


def test_interpolate_substitutes_known_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('AGENT_MODEL', 'claude-sonnet-5')

    assert (
        interpolate('model: ${AGENT_MODEL}', 'defaults.model')
        == 'model: claude-sonnet-5'
    )


def test_interpolate_raises_for_missing_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('MISSING_VAR', raising=False)

    with pytest.raises(
        MissingEnvironmentVariableError, match=r'defaults\.model.*MISSING_VAR'
    ):
        interpolate('${MISSING_VAR}', 'defaults.model')


# resolve_value_from()


def test_resolve_value_from_env_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('MEMORY_MCP_TOKEN', 'secret-token')

    resolved = resolve_value_from(
        {'env': 'MEMORY_MCP_TOKEN'}, 'mcp_servers.memory.auth'
    )

    assert resolved == 'secret-token'


def test_resolve_value_from_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('MEMORY_MCP_TOKEN', raising=False)

    with pytest.raises(
        MissingEnvironmentVariableError,
        match=r'mcp_servers\.memory\.auth.*MEMORY_MCP_TOKEN',
    ):
        resolve_value_from({'env': 'MEMORY_MCP_TOKEN'}, 'mcp_servers.memory.auth')


def test_resolve_value_from_file_succeeds(tmp_path: Path) -> None:
    secret_file = tmp_path / 'felo.token'
    secret_file.write_text('felo-secret\n')

    resolved = resolve_value_from({'file': str(secret_file)}, 'mcp_servers.felo.auth')

    assert resolved == 'felo-secret'


def test_resolve_value_from_file_missing_raises(tmp_path: Path) -> None:
    missing_file = tmp_path / 'missing.token'

    with pytest.raises(MissingConfigFileError, match=r'mcp_servers\.felo\.auth'):
        resolve_value_from({'file': str(missing_file)}, 'mcp_servers.felo.auth')


def test_resolve_value_from_rejects_both_env_and_file() -> None:
    with pytest.raises(InvalidValueReferenceError):
        resolve_value_from({'env': 'A', 'file': 'b'}, 'mcp_servers.memory.auth')


def test_resolve_value_from_rejects_unknown_shape() -> None:
    with pytest.raises(InvalidValueReferenceError):
        resolve_value_from({'bogus': 'A'}, 'mcp_servers.memory.auth')


# resolve_references()


def test_resolve_references_walks_nested_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('MEMORY_MCP_TOKEN', 'secret-token')
    monkeypatch.setenv('AGENT_MODEL', 'claude-sonnet-5')

    raw = {
        'defaults': {'model': '${AGENT_MODEL}'},
        'mcp_servers': {
            'memory': {
                'transport': 'http',
                'url': 'https://memory.example.internal/mcp',
                'auth': {'value_from': {'env': 'MEMORY_MCP_TOKEN'}},
            },
        },
        'agents': {
            'scout': {'tools': [{'name': 'websearch'}]},
        },
    }

    resolved = resolve_references(raw)

    assert resolved == {
        'defaults': {'model': 'claude-sonnet-5'},
        'mcp_servers': {
            'memory': {
                'transport': 'http',
                'url': 'https://memory.example.internal/mcp',
                'auth': 'secret-token',
            },
        },
        'agents': {
            'scout': {'tools': [{'name': 'websearch'}]},
        },
    }


def test_resolve_references_reports_field_path_for_missing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('MEMORY_MCP_TOKEN', raising=False)

    raw = {
        'mcp_servers': {'memory': {'auth': {'value_from': {'env': 'MEMORY_MCP_TOKEN'}}}}
    }

    with pytest.raises(
        MissingEnvironmentVariableError, match=r'mcp_servers\.memory\.auth'
    ):
        resolve_references(raw)


# load_config()


def test_load_config_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MissingConfigFileError):
        load_config(tmp_path / 'missing.yaml', build_section_registry())


def test_load_config_requires_agents_section(tmp_path: Path) -> None:
    config_file = tmp_path / 'huginn.yaml'
    config_file.write_text('defaults: {}\n')

    with pytest.raises(MissingSectionError, match='agents'):
        load_config(config_file, build_section_registry())


def test_load_config_rejects_unknown_section(tmp_path: Path) -> None:
    config_file = tmp_path / 'huginn.yaml'
    config_file.write_text('agents: {}\nbogus: {}\n')

    with pytest.raises(UnknownSectionError, match='bogus'):
        load_config(config_file, build_section_registry())


def test_load_config_defaults_optional_sections(tmp_path: Path) -> None:
    config_file = tmp_path / 'huginn.yaml'
    config_file.write_text(
        'agents:\n  scout:\n    description: x\n    instructions: y\n'
    )

    config = load_config(config_file, build_section_registry())

    assert config.section('defaults', DefaultsConfig).model is None
    assert config.section('mcp_servers', McpServersConfig).root == {}
    assert config.section('channels', ChannelsConfig).telegram is None


def test_load_config_resolves_env_and_value_from_then_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('MEMORY_MCP_TOKEN', 'secret-token')
    monkeypatch.setenv('AGENT_MODEL', 'claude-sonnet-5')

    config_file = tmp_path / 'huginn.yaml'
    config_file.write_text(
        'defaults:\n'
        '  model: ${AGENT_MODEL}\n'
        'mcp_servers:\n'
        '  memory:\n'
        '    transport: http\n'
        '    url: https://memory.example.internal/mcp\n'
        '    auth:\n'
        '      value_from: {env: MEMORY_MCP_TOKEN}\n'
        'agents:\n'
        '  scout:\n'
        '    description: General-purpose research agent\n'
        '    instructions: You speak for ${AGENT_MODEL}.\n'
        '    tools: [{name: websearch, params: {max_results: 5}}]\n'
        '    mcp: [{server: memory, allow: [recall_memory_notes]}]\n'
        'channels:\n'
        '  telegram:\n'
        "    bindings: {'12345': scout}\n"
        '    router: {fallback: scout}\n'
    )

    config = load_config(config_file, build_section_registry())

    agents = config.section('agents', AgentsConfig).root
    servers = config.section('mcp_servers', McpServersConfig).root
    channels = config.section('channels', ChannelsConfig)

    assert config.section('defaults', DefaultsConfig).model == 'claude-sonnet-5'
    assert servers['memory'].auth is not None
    assert servers['memory'].auth.get_secret_value() == 'secret-token'
    assert agents['scout'].instructions == 'You speak for claude-sonnet-5.'
    assert channels.telegram is not None
    assert channels.telegram.bindings['12345'] == 'scout'


# validate_agents() / validate_channels()


def test_validators_accept_a_fully_consistent_config() -> None:
    config = build_app_config(
        agents={
            'scout': build_agent(
                tools=(ToolRef(name='websearch'),),
                mcp=(McpRef(server='memory'),),
            ),
        },
        mcp_servers={
            'memory': McpServerConfig(
                transport='http',
                url=HttpUrl('https://memory.example.internal/mcp'),
            ),
        },
        channels=ChannelsConfig(
            telegram=ChannelConfig(
                bindings={'12345': 'scout'},
                router=ChannelRouterConfig(fallback='scout'),
            ),
        ),
    )

    validate_agents(config, build_tool_registry())
    validate_channels(config)


def test_validate_agents_rejects_unknown_tool() -> None:
    config = build_app_config(
        agents={'scout': build_agent(tools=(ToolRef(name='nonexistent'),))},
    )

    with pytest.raises(
        UnknownToolReferenceError, match=r'agents\.scout\.tools\[0\]\.name'
    ):
        validate_agents(config, build_tool_registry())


def test_validate_agents_rejects_unknown_mcp_server() -> None:
    config = build_app_config(
        agents={'scout': build_agent(mcp=(McpRef(server='ghost'),))},
    )

    with pytest.raises(
        UnknownMcpServerReferenceError, match=r'agents\.scout\.mcp\[0\]\.server'
    ):
        validate_agents(config, build_tool_registry())


def test_validate_agents_requires_exactly_one_instruction_source() -> None:
    both = build_agent(instructions='a', instructions_file='b')
    neither = build_agent(instructions=None)

    with pytest.raises(AmbiguousInstructionsError, match=r'agents\.scout'):
        validate_agents(build_app_config(agents={'scout': both}), build_tool_registry())

    with pytest.raises(AmbiguousInstructionsError, match=r'agents\.scout'):
        validate_agents(
            build_app_config(agents={'scout': neither}), build_tool_registry()
        )


def test_validate_channels_rejects_unknown_binding_agent() -> None:
    config = build_app_config(
        agents={'scout': build_agent()},
        channels=ChannelsConfig(telegram=ChannelConfig(bindings={'42': 'ghost'})),
    )

    with pytest.raises(UnknownAgentReferenceError, match='bindings'):
        validate_channels(config)


def test_validate_channels_rejects_unknown_router_fallback() -> None:
    config = build_app_config(
        agents={'scout': build_agent()},
        channels=ChannelsConfig(
            telegram=ChannelConfig(router=ChannelRouterConfig(fallback='ghost')),
        ),
    )

    with pytest.raises(UnknownAgentReferenceError, match='fallback'):
        validate_channels(config)
