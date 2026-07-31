import asyncio
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import Annotated

from app.core.config.registry import HuginnConfig
from app.features.channels.config import CHANNELS
from app.features.channels.exceptions import MissingWebhookURLError
from app.features.channels.registry import CommandRegistry
from lib.command import Argument, Exit, Option, Typer, echo, scoped
from lib.container import ContainerContract
from lib.http import HTTPClientContract
from lib.temporal import isoformat

from .config import TelegramChannelConfig
from .repositories import TelegramMenuRepository, TelegramWebhookRepository
from .types import WebhookStatus

cli = Typer(help='Administer telegram bots.', no_args_is_help=True)

_ALLOWED_UPDATES = ('message',)

_CHANNEL_ARGUMENT = Argument(
    help='Channel instance to act on; every telegram bot by default.',
)
_DROP_PENDING_OPTION = Option(
    help='Discard the updates telegram has queued for this bot.',
)


def _channels(
    config: HuginnConfig,
    name: str | None,
) -> Iterator[tuple[str, TelegramChannelConfig]]:
    configured = config.section(CHANNELS).root
    for instance, channel in sorted(configured.items()):
        if not isinstance(channel, TelegramChannelConfig):
            continue

        if name is None or name == instance:
            yield instance, channel


def _targets(
    config: HuginnConfig,
    name: str | None,
) -> Sequence[tuple[str, TelegramChannelConfig]]:
    targets = list(_channels(config, name))
    if not targets:
        raise LookupError(
            'No telegram channel is configured.'
            if name is None
            else f'No telegram channel named {name!r}.',
        )

    return targets


def _base_url(channel: TelegramChannelConfig, override: str | None) -> str:
    if override is not None:
        return override

    if channel.webhook_base_url is None:
        raise MissingWebhookURLError(
            'No webhook_base_url is configured; pass --url instead.',
        )

    return str(channel.webhook_base_url)


def _failure(status: WebhookStatus) -> str:
    when = None if status.failed_at is None else isoformat(status.failed_at)

    return f'{when or "(unknown time)"} {status.failure or "(no message)"}'


def _describe(status: WebhookStatus) -> Iterator[str]:
    yield f'  url: {status.url or "(unregistered)"}'
    yield f'  pending updates: {status.pending}'
    yield f'  allowed updates: {", ".join(status.allowed) or "(vendor default)"}'
    if status.connections is not None:
        yield f'  max connections: {status.connections}'
    if status.certificate:
        yield '  custom certificate: yes'
    if status.failed_at is not None or status.failure is not None:
        yield f'  last error: {_failure(status)}'


def _run[T](operation: Callable[[ContainerContract], Awaitable[T]]) -> T:
    async def execute() -> T:
        async with scoped() as scope:
            return await operation(scope)

    try:
        return asyncio.run(execute())
    except (LookupError, MissingWebhookURLError) as error:
        echo(f'{error}', err=True)
        raise Exit(1) from error


@cli.command('set-my-commands')
def set_my_commands(
    channel: Annotated[str | None, _CHANNEL_ARGUMENT] = None,
) -> None:
    async def publish(scope: ContainerContract) -> None:
        menu = [
            (command.name, command.description)
            for command in scope.resolve(CommandRegistry).listing()
        ]
        repository = TelegramMenuRepository(scope.resolve(HTTPClientContract))

        for instance, configured in _targets(scope.resolve(HuginnConfig), channel):
            await repository.publish(configured.token, menu)
            echo(f'{instance}: published {len(menu)} commands')

    _run(publish)


@cli.command('set-webhook')
def set_webhook(
    channel: Annotated[str | None, _CHANNEL_ARGUMENT] = None,
    url: Annotated[
        str | None,
        Option(help="Base URL to register; overrides the channel's webhook_base_url."),
    ] = None,
    allowed_updates: Annotated[
        list[str] | None,
        Option(help='Update kind to receive; repeatable.'),
    ] = None,
    drop_pending: Annotated[bool, _DROP_PENDING_OPTION] = False,
) -> None:
    async def register(scope: ContainerContract) -> None:
        repository = TelegramWebhookRepository(scope.resolve(HTTPClientContract))

        for instance, configured in _targets(scope.resolve(HuginnConfig), channel):
            registered = await repository.register(
                configured.token,
                _base_url(configured, url),
                configured.secret_token,
                allowed_updates or _ALLOWED_UPDATES,
                drop_pending_updates=drop_pending,
            )
            echo(f'{instance}: registered {registered}')

    _run(register)


@cli.command('delete-webhook')
def delete_webhook(
    channel: Annotated[str | None, _CHANNEL_ARGUMENT] = None,
    drop_pending: Annotated[bool, _DROP_PENDING_OPTION] = False,
) -> None:
    async def unregister(scope: ContainerContract) -> None:
        repository = TelegramWebhookRepository(scope.resolve(HTTPClientContract))

        for instance, configured in _targets(scope.resolve(HuginnConfig), channel):
            await repository.unregister(
                configured.token,
                drop_pending_updates=drop_pending,
            )
            echo(f'{instance}: unregistered')

    _run(unregister)


@cli.command('webhook-info')
def webhook_info(
    channel: Annotated[str | None, _CHANNEL_ARGUMENT] = None,
) -> None:
    async def inspect(scope: ContainerContract) -> None:
        repository = TelegramWebhookRepository(scope.resolve(HTTPClientContract))

        for instance, configured in _targets(scope.resolve(HuginnConfig), channel):
            status = await repository.inspect(configured.token)
            echo(f'{instance}:')
            for line in _describe(status):
                echo(line)

    _run(inspect)
