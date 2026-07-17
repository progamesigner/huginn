import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.core.application import App
from app.entrypoints.cli import COMMANDS
from app.entrypoints.extensions import EXTENSIONS
from lib.command import CommandManager, Typer

runner = CliRunner()

_VALID = """
agents:
  scout:
    description: Research agent
    instructions: You are scout, a secret persona.
    tools:
    - name: websearch
mcp_servers:
  muninn:
    transport: http
    url: https://memory.test
channels:
  telegram:
    bindings:
      '42': scout
"""


def build_cli() -> Typer:
    container = App().build(*App.providers, *EXTENSIONS, *COMMANDS)
    return container.resolve(CommandManager).app


def write_config(path: Path, text: str = _VALID) -> Path:
    path.write_text(text)
    return path


def test_validate_prints_summary(tmp_path: Path) -> None:
    path = write_config(tmp_path / 'huginn.yaml')

    result = runner.invoke(build_cli(), ['config', 'validate', str(path)])

    assert result.exit_code == 0
    assert 'scout' in result.output
    assert 'websearch' in result.output
    assert 'muninn' in result.output


def test_validate_never_echoes_instructions(tmp_path: Path) -> None:
    path = write_config(tmp_path / 'huginn.yaml')

    result = runner.invoke(build_cli(), ['config', 'validate', str(path)])

    assert 'secret persona' not in result.output


def test_validate_defaults_to_settings_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_config(tmp_path / 'huginn.yaml')
    monkeypatch.setenv('HUGINN_CONFIG_FILE', str(path))

    result = runner.invoke(build_cli(), ['config', 'validate'])

    assert result.exit_code == 0


def test_validate_reports_field_path_on_bad_reference(tmp_path: Path) -> None:
    bad = _VALID.replace('name: websearch', 'name: nonexistent')
    path = write_config(tmp_path / 'huginn.yaml', bad)

    result = runner.invoke(build_cli(), ['config', 'validate', str(path)])

    assert result.exit_code == 1
    assert 'agents.scout.tools[0].name' in result.stderr


def test_validate_fails_on_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(
        build_cli(), ['config', 'validate', str(tmp_path / 'missing.yaml')]
    )

    assert result.exit_code == 1


def test_cli_builds_without_ext_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('EXT_FELO_API_KEY', raising=False)
    path = write_config(tmp_path / 'huginn.yaml')

    result = runner.invoke(build_cli(), ['config', 'validate', str(path)])

    assert result.exit_code == 0


def test_schema_prints_json_schema() -> None:
    result = runner.invoke(build_cli(), ['config', 'schema'])

    assert result.exit_code == 0
    assert 'agents' in json.loads(result.output)['properties']
