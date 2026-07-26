from pathlib import Path

import pytest

from app.core.application import App
from app.core.providers.config import ConfigProvider
from app.core.providers.temporal import ClockProvider
from app.features.skills.exceptions import SkillsNotConfiguredError
from app.features.skills.providers import SkillsProvider
from app.features.skills.repositories import SkillRepository

_CONFIGURED = """
mcp_servers:
  memory:
    transport: http
    url: https://memory.test

skills:
  server: memory
  inject:
    agent: default
"""


def build(path: Path) -> App:
    return App().build(ConfigProvider(path=path), ClockProvider(), SkillsProvider())


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
    config.write_text('skills: {}\n')

    container = build(config)

    with pytest.raises(SkillsNotConfiguredError):
        container.resolve(SkillRepository)


def test_provider_binds_without_a_config_file(tmp_path: Path) -> None:
    container = build(tmp_path / 'does-not-exist.yaml')

    assert container.bound(SkillRepository)
