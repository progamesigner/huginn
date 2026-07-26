import json
from pathlib import Path
from typing import Annotated, Literal, override

import pytest
from pydantic import Field, HttpUrl, SecretStr, ValidationError

from app.core.config.exceptions import (
    InvalidValueReferenceError,
    MissingConfigFileError,
    MissingEnvironmentVariableError,
    MissingSectionError,
    UnknownSectionError,
)
from app.core.config.loader import (
    _interpolate,
    _resolve_references,
    _resolve_value_from,
    load_config,
)
from app.core.config.registry import HuginnConfig, SectionRegistry, SectionSpec
from app.features.agents.config import (
    AGENTS,
    DEFAULTS,
    AgentConfig,
    AgentConfigList,
    DefaultsConfig,
    MCPRef,
    ToolRef,
    _validate_agents,
    resolve_model,
)
from app.features.agents.config import build_sections as build_agent_sections
from app.features.agents.exceptions import (
    AmbiguousInstructionsError,
    MissingModelError,
    UnknownAgentReferenceError,
)
from app.features.agents.types import ToolOutput, ToolParams, TypedTool
from app.features.channels.config import (
    CHANNELS,
    ChannelConfigList,
    ChannelRouterConfig,
    _validate_channels,
)
from app.features.channels.config import build_sections as build_channel_sections
from app.features.channels.exceptions import DuplicateChannelSecretError
from app.features.channels.registry import ChannelRegistry
from app.features.channels.telegram.config import TelegramChannelConfig
from app.features.mcp.config import (
    MCP_SERVERS,
    MCPServerConfig,
    MCPServerConfigList,
    _validate_agent_mcp_refs,
)
from app.features.mcp.exceptions import UnknownMCPServerReferenceError
from app.features.tools.config import (
    TOOLS,
    ToolConfig,
    ToolConfigList,
    _validate_agent_tool_refs,
    _validate_tools,
)
from app.features.tools.config import build_sections as build_tool_sections
from app.features.tools.exceptions import (
    UnknownToolKindError,
    UnknownToolReferenceError,
)
from app.features.tools.registry import ToolRegistry
from app.features.tools.websearch.config import WebSearchToolConfig


class SearchParams(ToolParams):
    query: Annotated[str, Field()]
    max_results: Annotated[int, Field()] = 5


class WebSearchTool(TypedTool[SearchParams]):
    name = 'websearch'
    description = 'Fake tool for config loader tests'
    params = SearchParams

    @override
    async def run(self, params: SearchParams) -> ToolOutput:
        return 'ok'


class FakeParams(ToolParams):
    pass


class FakeTool(TypedTool[FakeParams]):
    name = 'faketool'
    description = 'Fake tool kind that lives outside features/tools/config.py'
    params = FakeParams

    @override
    async def run(self, params: FakeParams) -> ToolOutput:
        return 'ok'


class FakeToolConfig(ToolConfig):
    kind: Annotated[Literal['faketool'], Field()]
    setting: Annotated[str, Field()] = 'default'


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool, WebSearchToolConfig)
    return registry


def build_channel_registry() -> ChannelRegistry:
    registry = ChannelRegistry()
    registry.register('telegram', TelegramChannelConfig)
    return registry


def build_section_registry(tools: ToolRegistry | None = None) -> SectionRegistry:
    registry = SectionRegistry()
    tool_registry = tools or build_tool_registry()
    for spec in build_tool_sections(tool_registry):
        registry.register(spec)
    for spec in build_agent_sections():
        registry.register(spec)
    for spec in build_channel_sections(build_channel_registry()):
        registry.register(spec)
    registry.register(MCP_SERVERS)
    return registry


def build_agent(**overrides: object) -> AgentConfig:
    fields = {
        'description': 'General-purpose research agent',
        'instructions': 'You are scout.',
    }
    return AgentConfig.model_validate(fields | overrides)


def websearch_config() -> WebSearchToolConfig:
    return WebSearchToolConfig(kind='websearch', api_key=SecretStr('felo-key'))


def telegram_channel(**overrides: object) -> TelegramChannelConfig:
    fields = {
        'kind': 'telegram',
        'token': SecretStr('telegram-token'),
        'secret_token': SecretStr('telegram-secret'),
    }
    return TelegramChannelConfig.model_validate(fields | overrides)


def build_app_config(
    agents: dict[str, AgentConfig] | None = None,
    tools: dict[str, WebSearchToolConfig] | None = None,
    mcp_servers: dict[str, MCPServerConfig] | None = None,
    channels: ChannelConfigList | None = None,
    defaults: DefaultsConfig | None = None,
) -> HuginnConfig:
    return HuginnConfig(
        {
            'agents': AgentConfigList(agents or {}),
            'tools': ToolConfigList(tools or {}),
            'mcp_servers': MCPServerConfigList(mcp_servers or {}),
            'channels': channels or ChannelConfigList({}),
            'defaults': defaults or DefaultsConfig(model='test-model'),
        },
    )


# _interpolate()


def test_interpolate_substitutes_known_env_var() -> None:
    assert (
        _interpolate(
            'model: ${AGENT_MODEL}',
            'defaults.model',
            {'AGENT_MODEL': 'claude-sonnet-5'},
        )
        == 'model: claude-sonnet-5'
    )


def test_interpolate_raises_for_missing_env_var() -> None:
    with pytest.raises(
        MissingEnvironmentVariableError,
        match=r'defaults\.model.*MISSING_VAR',
    ):
        _interpolate('${MISSING_VAR}', 'defaults.model', {})


# _resolve_value_from()


def test_resolve_value_from_env_succeeds() -> None:
    resolved = _resolve_value_from(
        {'env': 'MEMORY_MCP_TOKEN'},
        'mcp_servers.memory.auth',
        {'MEMORY_MCP_TOKEN': 'secret-token'},
    )

    assert resolved == 'secret-token'


def test_resolve_value_from_env_missing_raises() -> None:
    with pytest.raises(
        MissingEnvironmentVariableError,
        match=r'mcp_servers\.memory\.auth.*MEMORY_MCP_TOKEN',
    ):
        _resolve_value_from({'env': 'MEMORY_MCP_TOKEN'}, 'mcp_servers.memory.auth', {})


def test_resolve_value_from_file_succeeds(tmp_path: Path) -> None:
    secret_file = tmp_path / 'felo.token'
    secret_file.write_text('felo-secret\n')

    resolved = _resolve_value_from(
        {'file': str(secret_file)},
        'tools.websearch.api_key',
        {},
    )

    assert resolved == 'felo-secret'


def test_resolve_value_from_file_missing_raises(tmp_path: Path) -> None:
    missing_file = tmp_path / 'missing.token'

    with pytest.raises(MissingConfigFileError, match=r'tools\.websearch\.api_key'):
        _resolve_value_from({'file': str(missing_file)}, 'tools.websearch.api_key', {})


def test_resolve_value_from_rejects_both_env_and_file() -> None:
    with pytest.raises(InvalidValueReferenceError):
        _resolve_value_from({'env': 'A', 'file': 'b'}, 'mcp_servers.memory.auth', {})


def test_resolve_value_from_rejects_unknown_shape() -> None:
    with pytest.raises(InvalidValueReferenceError):
        _resolve_value_from({'bogus': 'A'}, 'mcp_servers.memory.auth', {})


# _resolve_references()


def test_resolve_references_walks_nested_structure() -> None:
    raw = {
        'defaults': {'model': '${AGENT_MODEL}'},
        'tools': {
            'websearch': {
                'kind': 'websearch',
                'api_key': {'value_from': {'env': 'FELO_API_KEY'}},
            },
        },
        'agents': {
            'scout': {'tools': [{'name': 'websearch'}]},
        },
    }

    resolved = _resolve_references(
        raw,
        environment={
            'FELO_API_KEY': 'felo-secret',
            'AGENT_MODEL': 'claude-sonnet-5',
        },
    )

    assert resolved == {
        'defaults': {'model': 'claude-sonnet-5'},
        'tools': {
            'websearch': {
                'kind': 'websearch',
                'api_key': 'felo-secret',
            },
        },
        'agents': {
            'scout': {'tools': [{'name': 'websearch'}]},
        },
    }


def test_resolve_references_reports_field_path_for_missing_env() -> None:
    raw = {'tools': {'websearch': {'api_key': {'value_from': {'env': 'FELO_API_KEY'}}}}}

    with pytest.raises(
        MissingEnvironmentVariableError,
        match=r'tools\.websearch\.api_key',
    ):
        _resolve_references(raw, environment={})


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
        'agents:\n'
        '  scout:\n'
        '    description: x\n'
        '    instructions: y\n'
        '    model: test-model\n',
    )

    config = load_config(config_file, build_section_registry())

    assert config.section(DEFAULTS).model is None
    assert config.section(TOOLS).root == {}
    assert config.section(MCP_SERVERS).root == {}
    assert config.section(CHANNELS).root == {}


def test_load_config_resolves_env_and_value_from_then_validates(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / 'huginn.yaml'
    config_file.write_text(
        'defaults:\n'
        '  model: ${AGENT_MODEL}\n'
        'tools:\n'
        '  websearch:\n'
        '    kind: websearch\n'
        '    api_key:\n'
        '      value_from: {env: FELO_API_KEY}\n'
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
        '  support_bot:\n'
        '    kind: telegram\n'
        '    token: ${TELEGRAM_BOT_TOKEN}\n'
        '    secret_token: ${TELEGRAM_WEBHOOK_SECRET}\n'
        "    bindings: {'12345': scout}\n"
        '    router: {fallback: scout}\n',
    )

    config = load_config(
        config_file,
        build_section_registry(),
        {
            'FELO_API_KEY': 'felo-secret',
            'MEMORY_MCP_TOKEN': 'secret-token',
            'AGENT_MODEL': 'claude-sonnet-5',
            'TELEGRAM_BOT_TOKEN': 'telegram-token',
            'TELEGRAM_WEBHOOK_SECRET': 'telegram-secret',
        },
    )

    agents = config.section(AGENTS).root
    tools = config.section(TOOLS).root
    channels = config.section(CHANNELS)

    websearch = tools['websearch']
    assert isinstance(websearch, WebSearchToolConfig)

    assert config.section(DEFAULTS).model == 'claude-sonnet-5'
    assert websearch.api_key.get_secret_value() == 'felo-secret'
    assert agents['scout'].instructions == 'You speak for claude-sonnet-5.'
    assert channels.root['support_bot'].bindings['12345'] == 'scout'


def test_new_tool_kind_parses_and_schemas_without_touching_config_module(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    registry.register(WebSearchTool, WebSearchToolConfig)
    registry.register(FakeTool, FakeToolConfig)
    section_registry = build_section_registry(registry)

    config_file = tmp_path / 'huginn.yaml'
    config_file.write_text(
        'tools:\n'
        '  custom:\n'
        '    kind: faketool\n'
        '    setting: hi\n'
        'agents:\n'
        '  scout:\n'
        '    description: x\n'
        '    instructions: y\n'
        '    model: test-model\n'
        '    tools: [{name: custom}]\n',
    )

    config = load_config(config_file, section_registry)

    custom = config.section(TOOLS).root['custom']
    assert custom.kind == 'faketool'
    assert isinstance(custom, FakeToolConfig)
    assert 'faketool' in json.dumps(section_registry.schema())


def test_configless_tool_kind_parses_as_base_tool_config(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(WebSearchTool, WebSearchToolConfig)
    registry.register(FakeTool)
    section_registry = build_section_registry(registry)

    config_file = tmp_path / 'huginn.yaml'
    config_file.write_text(
        'tools:\n'
        '  custom:\n'
        '    kind: faketool\n'
        'agents:\n'
        '  scout:\n'
        '    description: x\n'
        '    instructions: y\n'
        '    model: test-model\n'
        '    tools: [{name: custom}]\n',
    )

    config = load_config(config_file, section_registry)

    custom = config.section(TOOLS).root['custom']
    assert type(custom) is ToolConfig
    assert custom.kind == 'faketool'


def test_unknown_tool_kind_fails_at_parse_listing_registered_tags(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / 'huginn.yaml'
    config_file.write_text(
        'tools:\n'
        '  mystery:\n'
        '    kind: bogus\n'
        'agents:\n'
        '  scout:\n'
        '    description: x\n'
        '    instructions: y\n',
    )

    with pytest.raises(ValidationError, match=r"'bogus'.*'websearch'"):
        load_config(config_file, build_section_registry())


# SectionSpec.resolved


def test_section_spec_resolves_concrete_model_to_itself() -> None:
    spec = SectionSpec('defaults', DefaultsConfig)

    assert spec.resolved is DefaultsConfig


def test_section_spec_resolves_thunk_lazily_and_once() -> None:
    calls = []

    def thunk() -> type[DefaultsConfig]:
        calls.append(1)
        return DefaultsConfig

    spec = SectionSpec('defaults', DefaultsConfig, thunk=thunk)
    assert calls == []

    assert spec.resolved is DefaultsConfig
    assert spec.resolved is DefaultsConfig
    assert calls == [1]


# _validate_tools() / _validate_agents() / _validate_channels()


def test_validators_accept_a_fully_consistent_config() -> None:
    config = build_app_config(
        agents={
            'scout': build_agent(
                tools=(ToolRef(name='websearch'),),
                mcp=(MCPRef(server='memory'),),
            ),
        },
        tools={'websearch': websearch_config()},
        mcp_servers={
            'memory': MCPServerConfig(
                transport='http',
                url=HttpUrl('https://memory.example.internal/mcp'),
            ),
        },
        channels=ChannelConfigList(
            {
                'support_bot': telegram_channel(
                    bindings={'12345': 'scout'},
                    router=ChannelRouterConfig(fallback='scout'),
                ),
            },
        ),
    )

    _validate_tools(config, build_tool_registry())
    _validate_agents(config)
    _validate_channels(config, build_channel_registry())


def test_validate_tools_rejects_unregistered_kind() -> None:
    config = build_app_config(tools={'websearch': websearch_config()})

    with pytest.raises(UnknownToolKindError, match=r'tools\.websearch\.kind'):
        _validate_tools(config, ToolRegistry())


def test_tools_section_rejects_unknown_tool_reference() -> None:
    config = build_app_config(
        agents={'scout': build_agent(tools=(ToolRef(name='nonexistent'),))},
        tools={'websearch': websearch_config()},
    )

    with pytest.raises(
        UnknownToolReferenceError,
        match=r'agents\.scout\.tools\[0\]\.name',
    ):
        _validate_agent_tool_refs(config, build_tool_registry())


def test_mcp_section_rejects_unknown_mcp_server() -> None:
    config = build_app_config(
        agents={'scout': build_agent(mcp=(MCPRef(server='ghost'),))},
    )

    with pytest.raises(
        UnknownMCPServerReferenceError,
        match=r'agents\.scout\.mcp\[0\]\.server',
    ):
        _validate_agent_mcp_refs(config)


def test_validate_agents_requires_exactly_one_instruction_source() -> None:
    both = build_agent(instructions='a', instructions_file='b')
    neither = build_agent(instructions=None)

    with pytest.raises(AmbiguousInstructionsError, match=r'agents\.scout'):
        _validate_agents(build_app_config(agents={'scout': both}))

    with pytest.raises(AmbiguousInstructionsError, match=r'agents\.scout'):
        _validate_agents(build_app_config(agents={'scout': neither}))


def test_resolve_model_prefers_explicit_over_default() -> None:
    assert resolve_model('claude-sonnet-5', 'gpt-5-mini') == 'claude-sonnet-5'


def test_resolve_model_falls_back_to_default() -> None:
    assert resolve_model(None, 'gpt-5-mini') == 'gpt-5-mini'


def test_resolve_model_raises_when_neither_is_set() -> None:
    with pytest.raises(MissingModelError):
        resolve_model(None, None)


def test_validate_agents_requires_an_effective_model() -> None:
    config = build_app_config(
        agents={'scout': build_agent(model=None)},
        defaults=DefaultsConfig(),
    )

    with pytest.raises(MissingModelError, match=r'agents\.scout\.model'):
        _validate_agents(config)


def test_validate_agents_accepts_agent_model_without_defaults() -> None:
    config = build_app_config(
        agents={'scout': build_agent(model='claude-sonnet-5')},
        defaults=DefaultsConfig(),
    )

    _validate_agents(config)


def test_validate_channels_requires_an_effective_router_model() -> None:
    config = build_app_config(
        agents={'scout': build_agent()},
        channels=ChannelConfigList(
            {
                'support_bot': telegram_channel(
                    router=ChannelRouterConfig(fallback='scout'),
                ),
            },
        ),
        defaults=DefaultsConfig(),
    )

    with pytest.raises(MissingModelError, match=r'channels.support_bot\.router\.model'):
        _validate_channels(config, build_channel_registry())


def test_validate_channels_rejects_unknown_binding_agent() -> None:
    config = build_app_config(
        agents={'scout': build_agent()},
        channels=ChannelConfigList(
            {'support_bot': telegram_channel(bindings={'42': 'ghost'})},
        ),
    )

    with pytest.raises(UnknownAgentReferenceError, match='bindings'):
        _validate_channels(config, build_channel_registry())


def test_validate_channels_rejects_unknown_router_fallback() -> None:
    config = build_app_config(
        agents={'scout': build_agent()},
        channels=ChannelConfigList(
            {
                'support_bot': telegram_channel(
                    router=ChannelRouterConfig(fallback='ghost'),
                ),
            },
        ),
    )

    with pytest.raises(UnknownAgentReferenceError, match='fallback'):
        _validate_channels(config, build_channel_registry())


def test_validate_channels_rejects_duplicate_secret_token() -> None:
    config = build_app_config(
        agents={'scout': build_agent()},
        channels=ChannelConfigList(
            {
                'support_bot': telegram_channel(agent='scout'),
                'research_bot': telegram_channel(agent='scout'),
            },
        ),
    )

    with pytest.raises(
        DuplicateChannelSecretError,
        match=r'research_bot\.secret_token',
    ):
        _validate_channels(config, build_channel_registry())
