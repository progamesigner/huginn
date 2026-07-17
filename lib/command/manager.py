from typing import override

from typer import Typer

from .contracts import CommandContract


class CommandManager(CommandContract):
    application: Typer

    def __init__(self) -> None:
        self.application = Typer(add_completion=False, no_args_is_help=True)

    @property
    @override
    def app(self) -> Typer:
        return self.application

    @override
    def mount(self, name: str, commands: Typer) -> None:
        self.application.add_typer(commands, name=name)
