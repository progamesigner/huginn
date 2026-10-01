import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from lib.command import Option, Typer, echo, resolve

from .services import RedisAuditor
from .types import AuditReport

cli = Typer(no_args_is_help=True, help='Audit external data stores.')


@cli.command('redis')
def redis(
    report: Annotated[Path | None, Option(help='Also write JSON here.')] = None,
) -> None:
    async def run() -> AuditReport:
        return await resolve(RedisAuditor).run()

    result = asyncio.run(run())
    output = json.dumps(asdict(result), indent=2, sort_keys=True)
    echo(output)
    if report is not None:
        report.write_text(f'{output}\n')
