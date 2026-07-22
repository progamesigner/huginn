from typer.testing import CliRunner

from app.core.application import App
from app.features.auditor.providers import AuditCommandProvider
from lib.command import CommandContract

runner = CliRunner()


def test_command_provider_mounts_audit_group() -> None:
    container = App().build(AuditCommandProvider())

    result = runner.invoke(container.resolve(CommandContract).app, ['audit', '--help'])

    assert result.exit_code == 0
    assert 'redis' in result.output
