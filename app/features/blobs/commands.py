import asyncio
from typing import Annotated

from lib.command import Option, Typer, echo, resolve
from lib.database import DatabaseContract
from lib.guid import GUIDContract
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
        clock = resolve(ClockContract)
        async with resolve(DatabaseContract).transaction() as transaction:
            return await BlobRepository(
                transaction,
                resolve(GUIDContract),
                clock,
            ).prune(
                clock.now(),
                resolve(BlobReferenceRegistry).for_session(transaction),
                dry_run=dry_run,
            )

    count = asyncio.run(run())
    action = 'Would prune' if dry_run else 'Pruned'
    echo(f'{action} {count} blob(s).')
