from pathlib import Path

import pytest

from app.core.application import App
from app.core.providers.config import ConfigProvider
from app.core.providers.temporal import ClockProvider
from app.features.agents.contracts import ToolSourceRegistry
from app.features.agents.providers import AgentProvider
from app.features.skills.exceptions import SkillsNotConfiguredError
from app.features.skills.providers import SkillsProvider
from app.features.skills.repositories import SkillRepository
from app.features.skills.tools import SkillToolSource

_CONFIGURED = """
defaults:
  model: gpt-5-mini

agents:
  scout:
    description: Research agent
    instructions: You are scout.

mcp_servers:
  memory:
    transport: http
    url: https://memory.test

skills:
  server: memory
  inject:
    agent: default
"""

_UNCONFIGURED = """
defaults:
  model: gpt-5-mini

agents:
  scout:
    description: Research agent
    instructions: You are scout.
"""


def build(path: Path) -> App:
    return App().build(
        AgentProvider(),
        ConfigProvider(path=path),
        ClockProvider(),
        SkillsProvider(),
    )


def test_repository_resolves_when_a_source_is_configured(tmp_path: Path) -> None:
    config = tmp_path / 'huginn.yaml'
    config.write_text(_CONFIGURED)

    container = build(config)

    assert isinstance(container.resolve(SkillRepository), SkillRepository)


def test_repository_is_shared(tmp_path: Path) -> None:
    config = tmp_path / 'huginn.yaml'
    config.write_text(_CONFIGURED)

    container = build(config)

    # NOTE: the TTL cache is worthless unless every caller gets the same instance
    assert container.resolve(SkillRepository) is container.resolve(SkillRepository)


def test_repository_reports_an_unconfigured_source_on_resolve(tmp_path: Path) -> None:
    config = tmp_path / 'huginn.yaml'
    config.write_text(_UNCONFIGURED)

    container = build(config)

    with pytest.raises(SkillsNotConfiguredError):
        container.resolve(SkillRepository)


def test_provider_binds_without_a_config_file(tmp_path: Path) -> None:
    container = build(tmp_path / 'does-not-exist.yaml')

    assert container.bound(SkillRepository)


def test_provider_contributes_a_tool_source(tmp_path: Path) -> None:
    config = tmp_path / 'huginn.yaml'
    config.write_text(_CONFIGURED)

    container = build(config)

    sources = container.resolve(ToolSourceRegistry).sources()

    assert any(isinstance(source, SkillToolSource) for source in sources)
