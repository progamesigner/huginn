import asyncio
from collections.abc import Awaitable, Callable, Iterator
from typing import Annotated

from app.core.config.registry import HuginnConfig
from app.features.channels.config import CHANNELS
from app.features.channels.registry import CommandRegistry
from lib.command import Argument, Exit, Typer, echo, scoped
from lib.container import ContainerContract
from lib.http import HTTPClientContract

from .config import TelegramChannelConfig
from .repositories import TelegramMenuRepository

cli = Typer(help='Administer telegram bots.', no_args_is_help=True)


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


def _run[T](operation: Callable[[ContainerContract], Awaitable[T]]) -> T:
    async def execute() -> T:
        async with scoped() as scope:
            return await operation(scope)

    return asyncio.run(execute())


@cli.command('set-my-commands')
def set_my_commands(
    channel: Annotated[
        str | None,
        Argument(help='Channel instance to publish to; every telegram bot by default.'),
    ] = None,
) -> None:
    async def publish(scope: ContainerContract) -> None:
        menu = [
            (command.name, command.description)
            for command in scope.resolve(CommandRegistry).listing()
        ]
        repository = TelegramMenuRepository(scope.resolve(HTTPClientContract))
        targets = list(_channels(scope.resolve(HuginnConfig), channel))
        if not targets:
            raise LookupError('no telegram channel matched')

        for instance, configured in targets:
            await repository.publish(configured.token, menu)
            echo(f'{instance}: published {len(menu)} commands')

    try:
        _run(publish)
    except LookupError as error:
        echo(f'No telegram channel named {channel!r}.', err=True)
        raise Exit(1) from error
