from pathlib import Path

import pytest

from app.core.application import App
from app.core.config.exceptions import MissingConfigFileError
from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.core.providers.settings import SettingsProvider
from app.core.settings import HuginnSettings
from app.features.agents.config import AGENTS
from app.features.agents.providers import AgentProvider
from tests.helpers.settings import make_settings

_MINIMAL = """
defaults:
  model: gpt-5-mini
agents:
  scout:
    description: Research agent
    instructions: You are scout.
"""


def write_config(path: Path) -> Path:
    path.write_text(_MINIMAL)
    return path


def build_app(
    path: Path | None = None,
    settings: HuginnSettings | None = None,
) -> App:
    return App().build(
        SettingsProvider(settings),
        ConfigProvider(path),
        AgentProvider(),
    )


def test_provider_loads_config_from_explicit_path(tmp_path: Path) -> None:
    path = write_config(tmp_path / 'huginn.yaml')
    app = build_app(path)

    config = app.resolve(HuginnConfig)

    assert 'scout' in config.section(AGENTS).root


def test_provider_defaults_to_settings_config_file(tmp_path: Path) -> None:
    path = write_config(tmp_path / 'huginn.yaml')
    app = build_app(settings=make_settings(config_file=path))

    config = app.resolve(HuginnConfig)

    assert 'scout' in config.section(AGENTS).root


def test_config_is_shared(tmp_path: Path) -> None:
    path = write_config(tmp_path / 'huginn.yaml')
    app = build_app(path)

    assert app.resolve(HuginnConfig) is app.resolve(HuginnConfig)


def test_config_loads_lazily(tmp_path: Path) -> None:
    app = build_app(tmp_path / 'missing.yaml')

    with pytest.raises(MissingConfigFileError):
        app.resolve(HuginnConfig)
