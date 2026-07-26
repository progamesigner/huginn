from typer.testing import CliRunner

from app.core.application import App
from app.features.auditor.providers import AuditCommandProvider, AuditorProvider
from app.features.auditor.services import RedisAuditor
from lib.command import CommandContract

runner = CliRunner()


def test_command_provider_mounts_audit_group() -> None:
    container = App().build(AuditCommandProvider())

    result = runner.invoke(container.resolve(CommandContract).app, ['audit', '--help'])

    assert result.exit_code == 0
    assert 'redis' in result.output


def test_auditor_provider_binds_the_auditor_without_the_command() -> None:
    container = App().build(AuditorProvider())

    assert container.bound(RedisAuditor)
    assert not container.bound(CommandContract)
