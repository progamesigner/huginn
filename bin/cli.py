from pydantic import ValidationError

from app.core.application import App
from app.core.providers.config import ConfigCommandProvider
from app.core.providers.settings import SettingsProvider
from app.features.auditor.providers import AuditCommandProvider
from app.features.identities.providers import IdentitiesCommandProvider
from app.providers.bootstrap import BootstrapProvider
from app.providers.features import FeaturesProvider
from lib.command import CommandContract, echo

COMMANDS = (
    AuditCommandProvider(),
    ConfigCommandProvider(),
    IdentitiesCommandProvider(),
)


def main() -> None:
    try:
        container = App().build(
            SettingsProvider(),
            BootstrapProvider(),
            FeaturesProvider(),
            *COMMANDS,
        )
    except ValidationError as error:
        echo(str(error), err=True)
        raise SystemExit(1) from error

    container.resolve(CommandContract).app()


if __name__ == '__main__':
    main()
