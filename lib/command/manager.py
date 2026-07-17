from typer import Typer


class CommandManager:
    app: Typer

    def __init__(self) -> None:
        self.app = Typer(add_completion=False, no_args_is_help=True)

    def mount(self, name: str, commands: Typer) -> None:
        self.app.add_typer(commands, name=name)
