from pathlib import Path

import pytest

from app.core.application import App
from app.core.config.registry import HuginnConfig, SectionRegistry
from app.features.agents.config import AGENTS
from app.features.agents.providers import AgentProvider
from app.features.agents.router import Router
from app.features.channels.providers import ChannelProvider
from app.features.tools.exceptions import InvalidToolOverlayError
from app.features.tools.providers import ToolProvider, ToolRegistryProvider
from app.features.tools.registry import ToolRegistry
from app.features.tools.websearch.repositories import WebSearchRepository
from app.providers.contributions import (
    AgentContributions,
    ChannelContributions,
    CoreContributions,
    ToolContributions,
)
from lib.concurrency import TaskSupervisor
from tests.helpers.settings import app_with, make_settings

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

_ROUTER_PROMPT_OVERLAY = """
defaults:
  model: gpt-5-mini
  router_prompt: Choose the agent best suited to the request.
agents:
  scout:
    description: Research agent
    instructions: You are scout.
"""


def test_tool_registry_provider_binds_registry() -> None:
    app = App().build(ToolRegistryProvider())

    assert app.bound(ToolRegistry)


def test_tool_provider_binds_repository() -> None:
    app = App().build(ToolProvider())

    assert app.bound(WebSearchRepository)


def test_tool_contributions_register_websearch_kind() -> None:
    app = App().build(ToolContributions())

    assert 'websearch' in app.resolve(ToolRegistry).tools


def test_tool_contributions_register_read_document_kind() -> None:
    app = App().build(ToolContributions())

    assert 'read_document' in app.resolve(ToolRegistry).tools


def test_tool_contributions_add_only_the_tools_section() -> None:
    app = App().build(ToolContributions())

    keys = {spec.key for spec in app.resolve(SectionRegistry)}

    assert keys == {'tools'}


def test_feature_providers_contribute_no_sections_themselves() -> None:
    app = App().build(AgentProvider(), ChannelProvider())

    assert {spec.key for spec in app.resolve(SectionRegistry)} == set()


def test_section_contributions_add_up_to_their_own_sections() -> None:
    app = App().build(
        AgentContributions(),
        ChannelContributions(),
        CoreContributions(),
    )

    keys = {spec.key for spec in app.resolve(SectionRegistry)}

    assert keys == {'agents', 'channels', 'defaults', 'providers', 'transcription'}


def test_agents_config_provider_accepts_valid_tool_overlay(tmp_path: Path) -> None:
    path = tmp_path / 'huginn.yaml'
    path.write_text(_VALID_OVERLAY)
    app = app_with(make_settings(config_file=path)).build(
        AgentContributions(),
        CoreContributions(),
        ToolContributions(),
    )

    config = app.resolve(HuginnConfig)

    assert 'scout' in config.section(AGENTS).root


def test_agents_config_provider_rejects_invalid_tool_overlay(tmp_path: Path) -> None:
    path = tmp_path / 'huginn.yaml'
    path.write_text(_INVALID_OVERLAY)
    app = app_with(make_settings(config_file=path)).build(
        AgentContributions(),
        CoreContributions(),
        ToolContributions(),
    )

    with pytest.raises(InvalidToolOverlayError):
        app.resolve(HuginnConfig)


def test_agent_provider_threads_a_configured_router_prompt_into_the_router(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'huginn.yaml'
    path.write_text(_ROUTER_PROMPT_OVERLAY)
    app = app_with(make_settings(config_file=path)).build(
        AgentContributions(),
        CoreContributions(),
    )

    router = app.resolve(Router)

    assert router.prompt == 'Choose the agent best suited to the request.'


def test_agent_provider_leaves_the_router_prompt_unset_without_configuration(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'huginn.yaml'
    path.write_text(_VALID_OVERLAY)
    app = app_with(make_settings(config_file=path)).build(
        AgentContributions(),
        CoreContributions(),
        ToolContributions(),
    )

    router = app.resolve(Router)

    assert router.prompt is None


async def test_one_router_keeps_the_shadows_and_drains_them_on_dispose(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'huginn.yaml'
    path.write_text(_VALID_OVERLAY)
    app = app_with(make_settings(config_file=path)).build(
        AgentContributions(),
        CoreContributions(),
        ToolContributions(),
    )

    router = app.resolve(Router)

    assert app.resolve(Router) is router
    assert isinstance(router.shadows, TaskSupervisor)
    await app.dispose()
