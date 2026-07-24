import pytest
from pydantic import ValidationError

from app.features.agents.config import (
    AgentConfig,
    AgentsConfig,
    DefaultsConfig,
    MCPRef,
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
                    MCPRef(
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


def test_tool_ref_defaults_to_no_params_or_inject() -> None:
    ref = ToolRef(name='websearch')

    assert ref.params == {}
    assert ref.inject == {}


def test_tool_ref_accepts_params_and_inject() -> None:
    ref = ToolRef(name='websearch', params={'max_results': 5}, inject={'region': 'tw'})

    assert ref.params == {'max_results': 5}
    assert ref.inject == {'region': 'tw'}


def test_mcp_ref_defaults_to_no_allowlist_or_injected_params() -> None:
    ref = MCPRef(server='memory')

    assert ref.allow == ()
    assert ref.inject == {}
