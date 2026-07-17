from pydantic import ValidationError

from app.core.application import App
from app.core.providers.config import ConfigCommandProvider
from app.providers.bootstrap import BootstrapProvider
from app.providers.features import FeaturesProvider
from lib.command import CommandManager, echo
from lib.container import ProviderContract

COMMANDS: tuple[ProviderContract, ...] = (ConfigCommandProvider(),)


def main() -> None:
    try:
        container = App().build(BootstrapProvider(), FeaturesProvider(), *COMMANDS)
    except ValidationError as error:
        echo(str(error), err=True)
        raise SystemExit(1) from error

    container.resolve(CommandManager).app()


if __name__ == '__main__':
    main()
