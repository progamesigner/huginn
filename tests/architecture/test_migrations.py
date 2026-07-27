import re

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.schema import CreateTable

from app.core.records import AppRecord
from tests.helpers.schemas import mapped_tables

# NOTE: the history check reads the revision files only, so it is marked per test
# rather than for the module and still runs where there is no database
_integration = pytest.mark.integration

# NOTE: mirrors _VERSION_TABLE in migrations/default/env.py, which cannot be imported
# because importing it runs the migrations
_VERSION_TABLE = 'migrations'
_CONSTRAINT = re.compile(r'CONSTRAINT (\S+) (FOREIGN KEY|PRIMARY KEY|UNIQUE|CHECK)')
_KINDS = {'FOREIGN KEY': 'f', 'PRIMARY KEY': 'p', 'UNIQUE': 'u', 'CHECK': 'c'}
# NOTE: cast contype because asyncpg hands PostgreSQL's "char" back as bytes
_CONSTRAINTS = """
select conrelid::regclass::text, conname, contype::text
from pg_constraint
where connamespace = 'public'::regnamespace and contype in ('c', 'f', 'p', 'u')
"""


def _drift(connection: Connection) -> list[object]:
    # NOTE: alembic compares types by default but not server defaults, and this has to
    # catch both; it deliberately ignores its own version table
    context = MigrationContext.configure(
        connection,
        opts={'version_table': _VERSION_TABLE, 'compare_server_default': True},
    )
    return compare_metadata(context, AppRecord.metadata)


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


async def _drifted(engine: AsyncEngine) -> list[object]:
    async with engine.connect() as connection:
        return await connection.run_sync(_drift)


async def _public_tables(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as connection:
        result = await connection.execute(
            text("select tablename from pg_tables where schemaname = 'public'"),
        )
        return {row[0] for row in result}


async def _live_constraints(
    engine: AsyncEngine,
    tables: frozenset[str],
) -> set[tuple[str, str, str]]:
    async with engine.connect() as connection:
        result = await connection.execute(text(_CONSTRAINTS))
        return {(row[0], row[1], row[2]) for row in result if row[0] in tables}


def _declared_constraints() -> set[tuple[str, str, str]]:
    # NOTE: read the names off the DDL SQLAlchemy would emit rather than off
    # constraint.name — only rendering makes the naming convention materialise
    return {
        (table.name, name.strip('"'), _KINDS[kind])
        for table in AppRecord.metadata.tables.values()
        for name, kind in _CONSTRAINT.findall(
            str(CreateTable(table).compile(dialect=postgresql.dialect())),
        )
    }


@_integration
async def test_declared_constraints_match_the_migrated_schema(
    database_url: str,
) -> None:
    tables = mapped_tables()
    engine = create_async_engine(database_url)
    try:
        await _drive(engine, 'up', 'default@head')
        live = await _live_constraints(engine, tables)
    finally:
        await engine.dispose()

    assert _declared_constraints() == live


@_integration
async def test_the_orm_matches_the_migrated_schema(database_url: str) -> None:
    assert mapped_tables()

    engine = create_async_engine(database_url)
    try:
        await _drive(engine, 'up', 'default@head')
        drift = await _drifted(engine)
    finally:
        await engine.dispose()

    assert not drift, 'the ORM and the migrated schema disagree:\n' + '\n'.join(
        repr(entry) for entry in drift
    )


@_integration
async def test_migration_upgrade_downgrade_upgrade_roundtrip(database_url: str) -> None:
    tables = mapped_tables()
    engine = create_async_engine(database_url)
    try:
        await _drive(engine, 'up', 'default@head')
        assert tables <= await _public_tables(engine)

        await _drive(engine, 'down', 'default@base')
        remaining = await _public_tables(engine)

        # NOTE: naming no table keeps this honest as schemas are added; the version
        # table is alembic's own and downgrading to base deliberately leaves it
        assert not (tables & remaining)
        assert remaining <= {_VERSION_TABLE}

        await _drive(engine, 'up', 'default@head')
        assert tables <= await _public_tables(engine)

        # NOTE: a downgrade that does not fully undo itself shows up here rather than
        # as a missing table -- the re-upgrade has to land back on the declared schema
        assert not await _drifted(engine)
    finally:
        await engine.dispose()


def test_migration_history_has_a_single_head() -> None:
    scripts = ScriptDirectory.from_config(Config(toml_file='pyproject.toml'))
    heads = scripts.get_heads()
    bases = scripts.get_bases()
    revisions = list(scripts.walk_revisions())

    assert revisions, 'no revision found; the check would pass vacuously'
    # NOTE: two heads mean two branches were authored against the same parent and
    # `upgrade head` becomes ambiguous; alembic only reports it when you run it
    assert len(heads) == 1, f'divergent migration branches: {sorted(heads)}'
    assert len(bases) == 1, f'disconnected migration roots: {sorted(bases)}'
