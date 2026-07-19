import asyncio

from alembic import context
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.application import App
from app.core.providers.database import DatabaseProvider
from lib.database import DatabaseContract, DatabaseManager

_VERSION_TABLE = 'migrations'

target_metadata = None


def _database() -> DatabaseManager:
    manager = App().build(DatabaseProvider()).resolve(DatabaseContract)
    if not isinstance(manager, DatabaseManager):
        raise TypeError('DatabaseProvider must bind a DatabaseManager')
    return manager


def _run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=_VERSION_TABLE,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    context.configure(
        url=_database().url,
        target_metadata=target_metadata,
        version_table=_VERSION_TABLE,
        literal_binds=True,
        dialect_opts={'paramstyle': 'named'},
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_migrations_online() -> None:
    engine = create_async_engine(_database().url, poolclass=pool.NullPool)
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
