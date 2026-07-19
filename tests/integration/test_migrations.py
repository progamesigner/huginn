import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

pytestmark = pytest.mark.integration

_TABLES = {'conversations', 'turns', 'parts'}


def _migrate(connection: Connection, direction: str, revision: str) -> None:
    # NOTE: env.py picks the live connection up from attributes and drives in-process
    config = Config(toml_file='pyproject.toml')
    config.attributes['connection'] = connection
    if direction == 'up':
        command.upgrade(config, revision)
    else:
        command.downgrade(config, revision)


async def _drive(engine: AsyncEngine, direction: str, revision: str) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(_migrate, direction, revision)


async def _public_tables(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as connection:
        result = await connection.execute(
            text("select tablename from pg_tables where schemaname = 'public'")
        )
        return {row[0] for row in result}


async def test_migration_upgrade_downgrade_upgrade_roundtrip(database_url: str) -> None:
    engine = create_async_engine(database_url)
    try:
        await _drive(engine, 'up', 'head')
        assert _TABLES <= await _public_tables(engine)

        await _drive(engine, 'down', 'base')
        assert not (_TABLES & await _public_tables(engine))

        await _drive(engine, 'up', 'head')
        assert _TABLES <= await _public_tables(engine)
    finally:
        await engine.dispose()
