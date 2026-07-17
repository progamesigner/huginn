from pathlib import Path
from typing import Annotated, override

import pytest
from pydantic import Field, HttpUrl

from app.core.config import (
    AgentConfig,
    ChannelConfig,
    ChannelRouterConfig,
    ChannelsConfig,
    HuginnConfig,
    McpRef,
    McpServerConfig,
    ToolRef,
    load_config,
)
from app.core.config.exceptions import (
    AmbiguousInstructionsError,
    InvalidValueReferenceError,
    MissingConfigFileError,
    MissingEnvironmentVariableError,
    UnknownAgentReferenceError,
    UnknownMcpServerReferenceError,
    UnknownToolReferenceError,
)
from app.core.config.loader import (
    interpolate,
    resolve_references,
    resolve_value_from,
    validate_references,
)
from app.core.tools import Tool, ToolOutput, ToolParams, ToolRegistry


class SearchParams(ToolParams):
    query: Annotated[str, Field()]


class WebSearchTool(Tool[SearchParams]):
    name = 'websearch'
    description = 'Fake tool for config loader tests'
    params = SearchParams

    @override
    async def run(self, params: SearchParams) -> ToolOutput:
        return 'ok'


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool)
    return registry


def build_agent(**overrides: object) -> AgentConfig:
    fields: dict[str, object] = {
        'description': 'General-purpose research agent',
        'instructions': 'You are scout.',
    }
    fields.update(overrides)
    return AgentConfig.model_validate(fields)


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
    monkeypatch.setenv('MUNINN_TOKEN', 'secret-token')

    resolved = resolve_value_from({'env': 'MUNINN_TOKEN'}, 'mcp_servers.muninn.auth')

    assert resolved == 'secret-token'


def test_resolve_value_from_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('MUNINN_TOKEN', raising=False)

    with pytest.raises(
        MissingEnvironmentVariableError,
        match=r'mcp_servers\.muninn\.auth.*MUNINN_TOKEN',
    ):
        resolve_value_from({'env': 'MUNINN_TOKEN'}, 'mcp_servers.muninn.auth')


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
        resolve_value_from({'env': 'A', 'file': 'b'}, 'mcp_servers.muninn.auth')


def test_resolve_value_from_rejects_unknown_shape() -> None:
    with pytest.raises(InvalidValueReferenceError):
        resolve_value_from({'bogus': 'A'}, 'mcp_servers.muninn.auth')


# resolve_references()


def test_resolve_references_walks_nested_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('MUNINN_TOKEN', 'secret-token')
    monkeypatch.setenv('AGENT_MODEL', 'claude-sonnet-5')

    raw = {
        'defaults': {'model': '${AGENT_MODEL}'},
        'mcp_servers': {
            'muninn': {
                'transport': 'http',
                'url': 'https://memory.progamesigner.dev',
                'auth': {'value_from': {'env': 'MUNINN_TOKEN'}},
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
            'muninn': {
                'transport': 'http',
                'url': 'https://memory.progamesigner.dev',
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
    monkeypatch.delenv('MUNINN_TOKEN', raising=False)

    raw = {'mcp_servers': {'muninn': {'auth': {'value_from': {'env': 'MUNINN_TOKEN'}}}}}

    with pytest.raises(
        MissingEnvironmentVariableError, match=r'mcp_servers\.muninn\.auth'
    ):
        resolve_references(raw)


# load_config()


def test_load_config_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MissingConfigFileError):
        load_config(tmp_path / 'missing.yaml', build_registry())


def test_load_config_resolves_env_and_value_from_then_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('MUNINN_TOKEN', 'secret-token')
    monkeypatch.setenv('AGENT_MODEL', 'claude-sonnet-5')

    config_file = tmp_path / 'huginn.yaml'
    config_file.write_text(
        'defaults:\n'
        '  model: ${AGENT_MODEL}\n'
        'mcp_servers:\n'
        '  muninn:\n'
        '    transport: http\n'
        '    url: https://memory.progamesigner.dev\n'
        '    auth:\n'
        '      value_from: {env: MUNINN_TOKEN}\n'
        'agents:\n'
        '  scout:\n'
        '    description: General-purpose research agent\n'
        '    instructions: You speak for ${AGENT_MODEL}.\n'
        '    tools: [{name: websearch, params: {max_results: 5}}]\n'
        '    mcp: [{server: muninn, allow: [recall_memory_notes]}]\n'
        'channels:\n'
        '  telegram:\n'
        "    bindings: {'12345': scout}\n"
        '    router: {fallback: scout}\n'
    )

    config = load_config(config_file, build_registry())

    assert config.defaults.model == 'claude-sonnet-5'
    assert config.mcp_servers['muninn'].auth is not None
    assert config.mcp_servers['muninn'].auth.get_secret_value() == 'secret-token'
    assert config.agents['scout'].instructions == 'You speak for claude-sonnet-5.'
    assert config.channels.telegram is not None
    assert config.channels.telegram.bindings['12345'] == 'scout'


# validate_references()


def test_validate_references_accepts_a_fully_consistent_config() -> None:
    config = HuginnConfig(
        agents={
            'scout': build_agent(
                tools=(ToolRef(name='websearch'),),
                mcp=(McpRef(server='muninn'),),
            ),
        },
        mcp_servers={
            'muninn': McpServerConfig(
                transport='http', url=HttpUrl('https://memory.example')
            ),
        },
        channels=ChannelsConfig(
            telegram=ChannelConfig(
                bindings={'12345': 'scout'},
                router=ChannelRouterConfig(fallback='scout'),
            ),
        ),
    )

    validate_references(config, build_registry())


def test_validate_references_rejects_conflicting_instructions(tmp_path: Path) -> None:
    instructions_file = tmp_path / 'scout.md'
    config = HuginnConfig(
        agents={
            'scout': build_agent(
                instructions='hi', instructions_file=str(instructions_file)
            ),
        },
    )

    with pytest.raises(AmbiguousInstructionsError, match=r'agents\.scout'):
        validate_references(config, build_registry())


def test_validate_references_rejects_missing_instructions() -> None:
    config = HuginnConfig(
        agents={'scout': build_agent(instructions=None)},
    )

    with pytest.raises(AmbiguousInstructionsError, match=r'agents\.scout'):
        validate_references(config, build_registry())


def test_validate_references_rejects_unknown_tool() -> None:
    config = HuginnConfig(
        agents={'scout': build_agent(tools=(ToolRef(name='not-registered'),))},
    )

    with pytest.raises(
        UnknownToolReferenceError, match=r'agents\.scout\.tools\[0\]\.name'
    ):
        validate_references(config, build_registry())


def test_validate_references_rejects_unknown_mcp_server() -> None:
    config = HuginnConfig(
        agents={'scout': build_agent(mcp=(McpRef(server='not-declared'),))},
    )

    with pytest.raises(
        UnknownMcpServerReferenceError, match=r'agents\.scout\.mcp\[0\]\.server'
    ):
        validate_references(config, build_registry())


def test_validate_references_rejects_unknown_binding_agent() -> None:
    config = HuginnConfig(
        agents={'scout': build_agent()},
        channels=ChannelsConfig(
            telegram=ChannelConfig(bindings={'12345': 'ghost'}),
        ),
    )

    with pytest.raises(
        UnknownAgentReferenceError, match=r'channels\.telegram\.bindings\.12345'
    ):
        validate_references(config, build_registry())


def test_validate_references_rejects_unknown_fallback_agent() -> None:
    config = HuginnConfig(
        agents={'scout': build_agent()},
        channels=ChannelsConfig(
            telegram=ChannelConfig(router=ChannelRouterConfig(fallback='ghost')),
        ),
    )

    with pytest.raises(
        UnknownAgentReferenceError, match=r'channels\.telegram\.router\.fallback'
    ):
        validate_references(config, build_registry())
