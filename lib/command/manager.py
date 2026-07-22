from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import override

from typer import Context, Typer

from lib.container import ContainerContract

from .contracts import CommandContract

_container: ContextVar[ContainerContract | None] = ContextVar('container', default=None)


def _current() -> ContainerContract:
    container = _container.get()
    if container is None:
        raise RuntimeError('command container is not configured')

    return container


class CommandManager(CommandContract):
    application: Typer

    def __init__(self, container: ContainerContract) -> None:
        def initialize(context: Context) -> None:
            token = _container.set(container)
            context.call_on_close(lambda: _container.reset(token))

        self.application = Typer(
            callback=initialize,
            add_completion=False,
            no_args_is_help=True,
        )

    @property
    @override
    def app(self) -> Typer:
        return self.application

    @override
    def mount(self, name: str, commands: Typer) -> None:
        self.application.add_typer(commands, name=name)


def resolve[T](binding: type[T]) -> T:
    return _current().resolve(binding)


@asynccontextmanager
async def scoped() -> AsyncIterator[ContainerContract]:
    async with _current().scope() as container:
        yield container
