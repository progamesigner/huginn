import asyncio
from typing import Annotated

from lib.command import Option, Typer, echo, scoped
from lib.temporal import ClockContract

from .contracts import BlobReferenceRegistry
from .repositories import BlobRepository

cli = Typer(no_args_is_help=True, help='Manage stored binary objects.')


@cli.command('prune')
def prune(
    dry_run: Annotated[
        bool,
        Option('--dry-run', help='Count expired, unreferenced blobs without deleting.'),
    ] = False,
) -> None:
    async def run() -> int:
        async with scoped() as container:
            repository = container.resolve(BlobRepository)
            references = container.resolve(BlobReferenceRegistry).for_session(
                repository.session,
            )
            async with repository.session.begin():
                return await repository.prune(
                    container.resolve(ClockContract).now(),
                    references,
                    dry_run=dry_run,
                )

    count = asyncio.run(run())
    action = 'Would prune' if dry_run else 'Pruned'
    echo(f'{action} {count} blob(s).')
