import asyncio
import re
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Annotated

from sqlalchemy.exc import IntegrityError

from lib.command import Argument, Exit, Typer, echo, scoped
from lib.container import ContainerContract

from .repositories import IdentityRepository
from .services import IdentityLinkingService

_TTL_PATTERN = re.compile(r'(?P<amount>[1-9][0-9]*)(?P<unit>[smhd])')
_TTL_UNITS = {
    's': timedelta(seconds=1),
    'm': timedelta(minutes=1),
    'h': timedelta(hours=1),
    'd': timedelta(days=1),
}

cli = Typer(
    no_args_is_help=True,
    help='Manage canonical users and identity links.',
)
users = Typer(no_args_is_help=True, help='Manage canonical users.')
cli.add_typer(users, name='user')


def _ttl(value: str) -> timedelta:
    match = _TTL_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError('ttl must be a positive duration such as 10m, 1h, or 1d')
    return int(match['amount']) * _TTL_UNITS[match['unit']]


def _run[T](
    operation: Callable[[ContainerContract], Awaitable[T]],
) -> T:
    async def execute() -> T:
        async with scoped() as scope:
            repository = scope.resolve(IdentityRepository)
            async with repository.session.begin():
                return await operation(scope)

    return asyncio.run(execute())


@users.command('create')
def create_user(
    handle: Annotated[str, Argument(help='Globally unique canonical handle.')],
    display_name: Annotated[str, Argument(help='User-facing display name.')],
) -> None:
    async def create(scope: ContainerContract) -> None:
        user = await scope.resolve(IdentityRepository).create_user(
            handle,
            display_name,
        )
        echo(f'Created {user.handle} ({user.id})')

    try:
        _run(create)
    except (IntegrityError, ValueError) as error:
        echo(f'Could not create user {handle!r}.', err=True)
        raise Exit(1) from error


@cli.command('invite')
def invite(
    user: Annotated[str, Argument(help='Canonical user handle.')],
    provider: Annotated[str, Argument(help='Identity provider.')],
    realm: Annotated[str, Argument(help='Stable provider realm key.')],
    ttl: Annotated[str, Argument(help='Validity duration, for example 10m.')],
    created_by: Annotated[str, Argument(help='Operator issuing the invite.')],
    bot_username: str | None = None,
) -> None:
    try:
        lifetime = _ttl(ttl)
    except ValueError as error:
        echo(str(error), err=True)
        raise Exit(1) from error

    async def issue(scope: ContainerContract) -> None:
        repository = scope.resolve(IdentityRepository)
        target = await repository.find_user(user)
        if target is None:
            raise LookupError('canonical user not found')
        issued = await scope.resolve(IdentityLinkingService).issue_invite(
            target.id,
            provider,
            realm,
            lifetime,
            created_by,
        )
        payload = f'link_{issued.token}'
        if bot_username is not None:
            bot = bot_username.removeprefix('@')
            echo(f'Deep link: https://t.me/{bot}?start={payload}')
        else:
            echo(f'Deep-link payload: {payload}')
        echo(f'Short code: {issued.short_code}')
        echo(f'Expires at: {issued.expires_at.isoformat()}')

    try:
        _run(issue)
    except LookupError as error:
        echo(f'User {user!r} was not found.', err=True)
        raise Exit(1) from error


@cli.command('list')
def list_identities(
    user: Annotated[str, Argument(help='Canonical user handle.')],
) -> None:
    async def list_for_user(scope: ContainerContract) -> None:
        repository = scope.resolve(IdentityRepository)
        target = await repository.find_user(user)
        if target is None:
            raise LookupError('canonical user not found')
        identities = await repository.list_external(target.id)
        if not identities:
            echo('No linked identities.')
            return
        for identity in identities:
            echo(
                f'{identity.id} {identity.provider} {identity.realm} '
                f'{identity.external_user_id}'
            )

    try:
        _run(list_for_user)
    except LookupError as error:
        echo(f'User {user!r} was not found.', err=True)
        raise Exit(1) from error


@cli.command('unlink')
def unlink(
    provider: Annotated[str, Argument(help='Identity provider.')],
    realm: Annotated[str, Argument(help='Stable provider realm key.')],
    external_user_id: Annotated[str, Argument(help='Provider user identifier.')],
) -> None:
    async def remove(scope: ContainerContract) -> None:
        repository = scope.resolve(IdentityRepository)
        identity = await repository.find_external(
            provider,
            realm,
            external_user_id,
        )
        if identity is None:
            raise LookupError('external identity not found')
        if not await scope.resolve(IdentityLinkingService).unlink(identity):
            raise LookupError('external identity not found')
        echo(f'Unlinked {provider}/{realm}/{external_user_id}.')

    try:
        _run(remove)
    except LookupError as error:
        echo('Linked identity was not found.', err=True)
        raise Exit(1) from error
