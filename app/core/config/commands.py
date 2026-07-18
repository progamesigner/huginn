import json
from pathlib import Path
from typing import Annotated

from pydantic import ValidationError

from app.core.settings import HuginnSettings
from lib.command import Argument, Exit, Typer, echo
from lib.container import ContainerContract

from .exceptions import ConfigError
from .loader import load_config
from .registry import SectionRegistry


def build_commands(container: ContainerContract) -> Typer:
    commands = Typer(no_args_is_help=True, help='Inspect and validate huginn.yaml.')

    @commands.command()
    def validate(
        path: Annotated[Path | None, Argument(help='Path to huginn.yaml.')] = None,
    ) -> None:
        registry = container.resolve(SectionRegistry)
        target = path or container.resolve(HuginnSettings).config_file
        try:
            config = load_config(target, registry)
        except (ConfigError, ValidationError) as error:
            echo(str(error), err=True)
            raise Exit(1) from error

        for spec in registry:
            if spec.describe is not None:
                for line in spec.describe(config):
                    echo(line)

    @commands.command()
    def schema() -> None:
        echo(json.dumps(container.resolve(SectionRegistry).schema(), indent=2))

    return commands
