import os
from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from app.core.settings import HuginnSettings

_PREFIX = 'HUGINN_'


@pytest.fixture(autouse=True)
def environment(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # NOTE: settings would otherwise read the ambient env, repo .env and huginn.yaml
    if request.node.get_closest_marker('integration'):
        return

    for name in list(os.environ):
        if name.startswith(_PREFIX):
            monkeypatch.delenv(name)

    config = SettingsConfigDict(HuginnSettings.model_config)
    config['env_file'] = str(tmp_path / '.env')

    monkeypatch.setattr(HuginnSettings, 'model_config', config)
    monkeypatch.setenv(f'{_PREFIX}CONFIG_FILE', str(tmp_path / 'huginn.yaml'))
