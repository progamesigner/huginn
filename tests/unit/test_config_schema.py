import pytest
from pydantic import HttpUrl, SecretStr, ValidationError

from app.core.config import (
    AgentConfig,
    ChannelConfig,
    ChannelRouterConfig,
    ChannelsConfig,
    DefaultsConfig,
    HuginnConfig,
    McpRef,
    McpServerConfig,
    ToolRef,
)


def test_huginn_config_accepts_full_example() -> None:
    config = HuginnConfig(
        defaults=DefaultsConfig(model='claude-sonnet-5'),
        mcp_servers={
            'memory': McpServerConfig(
                transport='http',
                url=HttpUrl('https://memory.example.internal/mcp'),
                auth=SecretStr('token'),
            ),
        },
        agents={
            'scout': AgentConfig(
                description='General-purpose research agent',
                model='claude-sonnet-5',
                instructions='You are scout.',
                tools=(ToolRef(name='websearch', params={'max_results': 5}),),
                mcp=(
                    McpRef(
                        server='memory',
                        allow=('recall_memory_notes',),
                        inject={'agent': 'default', 'user': 'swag'},
                    ),
                ),
            ),
        },
        channels=ChannelsConfig(
            telegram=ChannelConfig(
                bindings={'12345': 'scout'},
                router=ChannelRouterConfig(model='claude-haiku-4-5', fallback='scout'),
            ),
        ),
    )

    assert config.agents['scout'].description == 'General-purpose research agent'
    assert config.mcp_servers['memory'].transport == 'http'
    assert config.channels.telegram is not None
    assert config.channels.telegram.bindings['12345'] == 'scout'


def test_huginn_config_requires_agents() -> None:
    with pytest.raises(ValidationError, match='agents'):
        HuginnConfig.model_validate({})


def test_huginn_config_rejects_unknown_top_level_field() -> None:
    with pytest.raises(ValidationError):
        HuginnConfig.model_validate({'agents': {}, 'bogus': True})


def test_huginn_config_defaults_to_no_servers_and_no_channels() -> None:
    config = HuginnConfig.model_validate({'agents': {}})

    assert config.mcp_servers == {}
    assert config.channels.telegram is None
    assert config.defaults.model is None


def test_huginn_config_is_frozen() -> None:
    config = HuginnConfig.model_validate({'agents': {}})
    field = 'agents'

    with pytest.raises(ValidationError):
        setattr(config, field, {})


def test_agent_config_requires_description() -> None:
    with pytest.raises(ValidationError, match='description'):
        AgentConfig.model_validate({})


def test_agent_config_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        AgentConfig.model_validate({'description': 'x', 'bogus': True})


def test_agent_config_defaults_have_no_tools_or_mcp_or_instructions() -> None:
    agent = AgentConfig(description='x')

    assert agent.tools == ()
    assert agent.mcp == ()
    assert agent.instructions is None
    assert agent.instructions_file is None


def test_mcp_server_config_requires_transport_and_url() -> None:
    with pytest.raises(ValidationError, match='transport'):
        McpServerConfig.model_validate({'url': 'https://memory.example.internal/mcp'})

    with pytest.raises(ValidationError, match='url'):
        McpServerConfig.model_validate({'transport': 'http'})


def test_mcp_server_config_rejects_unknown_transport() -> None:
    with pytest.raises(ValidationError):
        McpServerConfig.model_validate(
            {'transport': 'stdio', 'url': 'https://memory.example.internal/mcp'}
        )


def test_tool_ref_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ToolRef.model_validate({'name': 'websearch', 'bogus': True})


def test_mcp_ref_defaults_to_no_allowlist_or_injected_params() -> None:
    ref = McpRef(server='memory')

    assert ref.allow == ()
    assert ref.inject == {}


def test_channel_router_config_requires_fallback() -> None:
    with pytest.raises(ValidationError, match='fallback'):
        ChannelRouterConfig.model_validate({})


def test_channels_config_rejects_unknown_channel_name() -> None:
    with pytest.raises(ValidationError):
        ChannelsConfig.model_validate({'discord': {}})
