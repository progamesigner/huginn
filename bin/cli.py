from pydantic import ValidationError

from app.core.application import App
from app.core.providers.bootstrap import BootstrapProvider
from app.core.providers.settings import SettingsProvider
from app.providers import HuginnCommandProvider, HuginnProvider
from lib.command import CommandContract, echo


def main() -> None:
    try:
        container = App().build(
            SettingsProvider(),
            BootstrapProvider(),
            HuginnProvider(),
            HuginnCommandProvider(),
        )
    except ValidationError as error:
        echo(str(error), err=True)
        raise SystemExit(1) from error

    container.resolve(CommandContract).app()


if __name__ == '__main__':
    main()
