from pydantic import ValidationError

from app.core.application import App
from lib.command import CommandManager, echo
from lib.container import ProviderContract

from .extensions import EXTENSIONS

COMMANDS: tuple[ProviderContract, ...] = ()


def main() -> None:
    try:
        container = App().build(*App.providers, *EXTENSIONS, *COMMANDS)
    except ValidationError as error:
        echo(str(error), err=True)
        raise SystemExit(1) from error

    container.resolve(CommandManager).app()


if __name__ == '__main__':
    main()
