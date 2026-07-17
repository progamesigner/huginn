from pathlib import Path

import pytest

from app.core.application import App
from app.core.config import HuginnConfig, MissingConfigFileError
from app.core.providers.config import ConfigProvider

_MINIMAL = """
agents:
  scout:
    description: Research agent
    instructions: You are scout.
"""


def write_config(path: Path) -> Path:
    path.write_text(_MINIMAL)
    return path


def test_provider_loads_config_from_explicit_path(tmp_path: Path) -> None:
    path = write_config(tmp_path / 'huginn.yaml')
    app = App().build(ConfigProvider(path))

    config = app.resolve(HuginnConfig)

    assert 'scout' in config.agents


def test_provider_defaults_to_settings_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_config(tmp_path / 'huginn.yaml')
    monkeypatch.setenv('HUGINN_CONFIG_FILE', str(path))
    app = App().build(ConfigProvider())

    config = app.resolve(HuginnConfig)

    assert 'scout' in config.agents


def test_config_is_shared(tmp_path: Path) -> None:
    path = write_config(tmp_path / 'huginn.yaml')
    app = App().build(ConfigProvider(path))

    assert app.resolve(HuginnConfig) is app.resolve(HuginnConfig)


def test_config_loads_lazily(tmp_path: Path) -> None:
    app = App().build(ConfigProvider(tmp_path / 'missing.yaml'))

    with pytest.raises(MissingConfigFileError):
        app.resolve(HuginnConfig)
