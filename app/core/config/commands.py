import json
from pathlib import Path
from typing import Annotated

from pydantic import ValidationError

from app.core.settings import HuginnSettings
from lib.command import Argument, Exit, Typer, echo, resolve

from .exceptions import ConfigError
from .loader import load_config
from .registry import SectionRegistry

cli = Typer(no_args_is_help=True, help='Inspect and validate huginn.yaml.')


@cli.command('validate')
def validate(
    path: Annotated[Path | None, Argument(help='Path to huginn.yaml.')] = None,
) -> None:
    registry = resolve(SectionRegistry)
    target = path or resolve(HuginnSettings).config_file
    try:
        config = load_config(target, registry)
    except (ConfigError, ValidationError) as error:
        echo(str(error), err=True)
        raise Exit(1) from error

    for spec in registry:
        if spec.describe is not None:
            for line in spec.describe(config):
                echo(line)


@cli.command('schema')
def schema() -> None:
    registry = resolve(SectionRegistry)
    echo(json.dumps(registry.schema(), indent=2))
