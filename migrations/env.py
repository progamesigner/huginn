import asyncio

from alembic import context, util
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.application import App
from app.core.providers.database import DatabaseProvider
from lib.database import DatabaseContract

target_metadata = None


def _parse() -> tuple[str | None, str]:
    revision = getattr(context.config.cmd_opts, 'revision', None)
    domain = context.config.attributes.get('migration_domain')
    if domain is None and isinstance(revision, str) and '@' in revision:
        domain = revision.rpartition('@')[0].rpartition(':')[2]
    match domain:
        case None | 'default':
            return None, 'public'
        case _:
            raise util.CommandError(f'unknown or ambiguous migration domain {domain!r}')


def _url(instance: str | None) -> str:
    return App().build(DatabaseProvider()).resolve(DatabaseContract).url(instance)


def _run_migrations(
    connection: Connection | None,
    /,
    schema: str,
    url: str | None,
) -> None:
    context.configure(
        connection=connection,
        url=url,
        dialect_opts={'paramstyle': 'named'},
        literal_binds=url is not None,
        target_metadata=target_metadata,
        version_table='migrations',
        version_table_schema=schema,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_migrations_online(schema: str, url: str) -> None:
    engine = create_async_engine(url, poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_run_migrations, schema=schema, url=None)
    await engine.dispose()


instance, schema = _parse()
if context.is_offline_mode():
    _run_migrations(None, schema=schema, url=_url(instance))
elif (connection := context.config.attributes.get('connection')) is not None:
    _run_migrations(connection, schema=schema, url=None)
else:
    asyncio.run(_run_migrations_online(schema, _url(instance)))
