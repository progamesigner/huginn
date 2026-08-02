from pathlib import Path

import pytest

from app.core.application import App
from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.providers.config import ConfigProvider
from app.features.agents.config import AGENTS
from app.features.agents.providers import AgentProvider
from app.features.channels.providers import ChannelProvider
from app.features.tools.exceptions import InvalidToolOverlayError
from app.features.tools.providers import (
    ToolProvider,
    ToolRegistryProvider,
    ToolsConfigProvider,
)
from app.features.tools.registry import ToolRegistry
from app.features.tools.websearch.repositories import WebSearchRepository

_VALID_OVERLAY = """
defaults:
  model: gpt-5-mini
tools:
  websearch:
    kind: websearch
    api_key: felo-key
agents:
  scout:
    description: Research agent
    instructions: You are scout.
    tools:
    - name: websearch
      params:
        max_results: 3
"""

_INVALID_OVERLAY = """
defaults:
  model: gpt-5-mini
tools:
  websearch:
    kind: websearch
    api_key: felo-key
agents:
  scout:
    description: Research agent
    instructions: You are scout.
    tools:
    - name: websearch
      params:
        bogus: 1
"""


def test_tool_registry_provider_binds_registry() -> None:
    app = App().build(ToolRegistryProvider())

    assert app.bound(ToolRegistry)


def test_tool_provider_binds_repository() -> None:
    app = App().build(ToolProvider())

    assert app.bound(WebSearchRepository)


def test_tool_provider_registers_websearch_kind() -> None:
    app = App().build(ToolProvider())

    assert 'websearch' in app.resolve(ToolRegistry).tools


def test_tools_config_provider_contributes_tools_on_top_of_agents() -> None:
    app = App().build(ToolsConfigProvider())

    keys = {spec.key for spec in app.resolve(SectionRegistry)}

    assert keys == {'agents', 'defaults', 'providers', 'tools'}


def test_agent_provider_contributes_only_its_own_sections() -> None:
    app = App().build(AgentProvider(), ChannelProvider())

    keys = {spec.key for spec in app.resolve(SectionRegistry)}

    assert keys == {'agents', 'channels', 'defaults', 'providers', 'transcription'}


def test_agents_config_provider_accepts_valid_tool_overlay(tmp_path: Path) -> None:
    path = tmp_path / 'huginn.yaml'
    path.write_text(_VALID_OVERLAY)
    app = App().build(ConfigProvider(path), ToolProvider(), AgentProvider())

    config = app.resolve(HuginnConfig)

    assert 'scout' in config.section(AGENTS).root


def test_agents_config_provider_rejects_invalid_tool_overlay(tmp_path: Path) -> None:
    path = tmp_path / 'huginn.yaml'
    path.write_text(_INVALID_OVERLAY)
    app = App().build(ConfigProvider(path), ToolProvider(), AgentProvider())

    with pytest.raises(InvalidToolOverlayError):
        app.resolve(HuginnConfig)
