import pytest
from pydantic import ValidationError

from app.features.agents.config import (
    AgentConfig,
    AgentsConfig,
    ChannelConfig,
    ChannelRouterConfig,
    ChannelsConfig,
    DefaultsConfig,
    McpRef,
    ToolRef,
)


def test_agents_config_accepts_full_example() -> None:
    agents = AgentsConfig(
        {
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
        }
    )

    assert agents.root['scout'].description == 'General-purpose research agent'
    assert agents.root['scout'].tools[0].name == 'websearch'


def test_defaults_config_defaults_to_no_model() -> None:
    assert DefaultsConfig().model is None


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


def test_agent_config_is_frozen() -> None:
    agent = AgentConfig(description='x')
    field = 'description'

    with pytest.raises(ValidationError):
        setattr(agent, field, 'y')


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


def test_channel_config_defaults_to_no_bindings_or_router() -> None:
    channel = ChannelConfig()

    assert channel.bindings == {}
    assert channel.router is None
