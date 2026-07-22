import asyncio

from alembic import context, util
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.application import App
from app.core.providers.database import DatabaseProvider
from lib.database import DatabaseContract, DatabaseManager

type ConnectionName = str | None
type MigrationDomain = tuple[ConnectionName, str]

_DOMAINS = {
    'default': (None, 'public'),
}
_VERSION_TABLE = 'migrations'

target_metadata = None


def _database() -> DatabaseManager:
    manager = App().build(DatabaseProvider()).resolve(DatabaseContract)
    if not isinstance(manager, DatabaseManager):
        raise TypeError('DatabaseProvider must bind a DatabaseManager')
    return manager


def _domain() -> str:
    revision = getattr(context.config.cmd_opts, 'revision', None)
    domain = context.config.attributes.get('migration_domain')
    if domain is None and isinstance(revision, str) and '@' in revision:
        domain = revision.rpartition('@')[0].rpartition(':')[2]
    if domain is None and len(_DOMAINS) == 1:
        return next(iter(_DOMAINS))
    if not isinstance(domain, str) or domain not in _DOMAINS:
        raise util.CommandError(f'unknown or ambiguous migration domain {domain!r}')
    return domain


def _parse() -> MigrationDomain:
    return _DOMAINS[_domain()]


def _run_migrations(connection: Connection) -> None:
    _, schema = _parse()
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=_VERSION_TABLE,
        version_table_schema=schema,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    instance, schema = _parse()
    context.configure(
        url=_database().url(instance),
        target_metadata=target_metadata,
        version_table=_VERSION_TABLE,
        version_table_schema=schema,
        include_schemas=True,
        literal_binds=True,
        dialect_opts={'paramstyle': 'named'},
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_migrations_online() -> None:
    instance, _ = _parse()
    engine = create_async_engine(_database().url(instance), poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    connection = context.config.attributes.get('connection')
    if connection is not None:
        _run_migrations(connection)
        return

    asyncio.run(_run_migrations_online())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
