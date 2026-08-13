import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from app.core.application import App
from app.core.providers.bootstrap import BootstrapProvider
from app.core.providers.settings import SettingsProvider
from app.core.settings import HuginnSettings
from app.providers import HuginnCommandProvider, HuginnProvider
from lib.command import CommandContract
from tests.helpers.settings import make_settings

runner = CliRunner()

_VALID = """
defaults:
  model: gpt-5-mini
tools:
  websearch:
    kind: websearch
    api_key: felo-key
agents:
  scout:
    description: Research agent
    instructions: You are scout, a secret persona.
    tools:
    - name: websearch
mcp_servers:
  memory:
    transport: http
    url: https://memory.test
channels:
  support_bot:
    kind: telegram
    token: telegram-token
    secret_token: telegram-secret
    agent: scout
"""


def invoke_cli(
    arguments: list[str],
    settings: HuginnSettings | None = None,
) -> Result:
    container = App().build(
        SettingsProvider(settings),
        BootstrapProvider(),
        HuginnProvider(),
        HuginnCommandProvider(),
    )
    return runner.invoke(container.resolve(CommandContract).app, arguments)


def write_config(path: Path, text: str = _VALID) -> Path:
    path.write_text(text)
    return path


def test_validate_prints_summary(tmp_path: Path) -> None:
    path = write_config(tmp_path / 'huginn.yaml')

    result = invoke_cli(['config', 'validate', str(path)])

    assert result.exit_code == 0
    assert 'scout' in result.output
    assert 'websearch' in result.output
    assert 'memory' in result.output


def test_validate_never_echoes_instructions(tmp_path: Path) -> None:
    path = write_config(tmp_path / 'huginn.yaml')

    result = invoke_cli(['config', 'validate', str(path)])

    assert 'secret persona' not in result.output


def test_validate_defaults_to_settings_config_file(tmp_path: Path) -> None:
    path = write_config(tmp_path / 'huginn.yaml')

    result = invoke_cli(
        ['config', 'validate'],
        make_settings(config_file=path),
    )

    assert result.exit_code == 0


def test_validate_reports_field_path_on_bad_reference(tmp_path: Path) -> None:
    bad = _VALID.replace('name: websearch', 'name: nonexistent')
    path = write_config(tmp_path / 'huginn.yaml', bad)

    result = invoke_cli(['config', 'validate', str(path)])

    assert result.exit_code == 1
    assert 'agents.scout.tools[0].name' in result.stderr


def test_validate_fails_on_missing_file(tmp_path: Path) -> None:
    result = invoke_cli(['config', 'validate', str(tmp_path / 'missing.yaml')])

    assert result.exit_code == 1


def test_cli_builds_without_ext_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('EXT_FELO_API_KEY', raising=False)
    path = write_config(tmp_path / 'huginn.yaml')

    result = invoke_cli(['config', 'validate', str(path)])

    assert result.exit_code == 0


def test_schema_prints_json_schema() -> None:
    result = invoke_cli(['config', 'schema'])

    assert result.exit_code == 0
    assert 'agents' in json.loads(result.output)['properties']
