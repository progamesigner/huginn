from datetime import UTC, datetime
from typing import override
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from typer.testing import CliRunner, Result

from app.core.application import App
from app.core.providers.command import CommandProvider
from app.features.identities import commands as identities
from app.features.identities.commands import cli
from app.features.identities.types import (
    ExternalIdentity,
    IssuedIdentityLinkInvite,
    User,
)
from lib.command import CommandContract
from lib.container import ContainerContract, ProviderContract
from lib.database import DatabaseContract
from lib.guid import GUIDContract
from lib.temporal import ClockContract

runner = CliRunner()
_USER = User(UUID(int=1), 'alice', 'Alice')
_IDENTITY = ExternalIdentity(UUID(int=2), _USER.id, 'telegram', 'primary', '42')


class FakeIdentitiesProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        transaction = MagicMock()
        transaction.__aenter__ = AsyncMock()
        transaction.__aexit__ = AsyncMock(return_value=False)
        database = MagicMock(spec=DatabaseContract)
        database.transaction.return_value = transaction
        container.bind(DatabaseContract, instance=database)
        container.bind(GUIDContract, instance=MagicMock(spec=GUIDContract))
        container.bind(ClockContract, instance=MagicMock(spec=ClockContract))


def dependencies() -> tuple[MagicMock, MagicMock]:
    repository = MagicMock()
    repository.create_user = AsyncMock(return_value=_USER)
    repository.find_user = AsyncMock(return_value=_USER)
    repository.list_external = AsyncMock(return_value=(_IDENTITY,))
    repository.find_external = AsyncMock(return_value=_IDENTITY)
    service = MagicMock()
    service.issue_invite = AsyncMock(
        return_value=IssuedIdentityLinkInvite(
            'opaque-token',
            'ABCD-EFGH-IJKL-MNOP-QRST-UVWX-YZ12-3456',
            datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )
    service.unlink = AsyncMock(return_value=True)
    return repository, service


def invoke(
    repository: MagicMock,
    service: MagicMock,
    arguments: list[str],
) -> Result:
    container = App().build(CommandProvider(), FakeIdentitiesProvider())
    commands = container.resolve(CommandContract)
    commands.mount('identities', cli)
    with (
        patch.object(identities, 'IdentityRepository', return_value=repository),
        patch.object(identities, '_linking', return_value=service),
    ):
        return runner.invoke(commands.app, ['identities', *arguments])


def test_user_create() -> None:
    repository, service = dependencies()

    result = invoke(
        repository,
        service,
        ['user', 'create', 'Alice', 'Alice Example'],
    )

    assert result.exit_code == 0
    repository.create_user.assert_awaited_once_with('Alice', 'Alice Example')
    assert 'Created alice' in result.output


def test_invite_prints_deep_link_and_short_code() -> None:
    repository, service = dependencies()

    result = invoke(
        repository,
        service,
        [
            'invite',
            'alice',
            'telegram',
            'primary',
            '10m',
            'operator',
            '--bot-username',
            '@huginn_bot',
        ],
    )

    assert result.exit_code == 0
    assert 'https://t.me/huginn_bot?start=link_opaque-token' in result.output
    assert 'Short code:' in result.output
    assert service.issue_invite.await_args.args[0] == _USER.id


def test_list_prints_linked_identities() -> None:
    repository, service = dependencies()

    result = invoke(repository, service, ['list', 'alice'])

    assert result.exit_code == 0
    assert f'{_IDENTITY.id} telegram primary 42' in result.output


def test_unlink_uses_service() -> None:
    repository, service = dependencies()

    result = invoke(
        repository,
        service,
        ['unlink', 'telegram', 'primary', '42'],
    )

    assert result.exit_code == 0
    service.unlink.assert_awaited_once_with(_IDENTITY)


def test_invite_rejects_invalid_ttl_without_issuing() -> None:
    repository, service = dependencies()

    result = invoke(
        repository,
        service,
        ['invite', 'alice', 'telegram', 'primary', 'soon', 'operator'],
    )

    assert result.exit_code == 1
    assert 'ttl must be a positive duration' in result.stderr
    service.issue_invite.assert_not_awaited()
