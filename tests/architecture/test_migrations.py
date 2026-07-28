import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import MetaData, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.records import AppRecord
from tests.helpers.schemas import mapped_tables

type Constraint = tuple[str, str, str, str]

# NOTE: the history check reads the revision files only, so it is marked per test
# rather than for the module and still runs where there is no database
_integration = pytest.mark.integration

# NOTE: mirrors _VERSION_TABLE in migrations/default/env.py, which cannot be imported
# because importing it runs the migrations
_VERSION_TABLE = 'migrations'
# Alembic compares a CHECK constraint's name but never its expression, so a rewritten
# predicate is invisible to autogenerate. Rather than diff expression text -- which
# PostgreSQL rewrites on the way in, casts and parentheses and all -- build the declared
# model into a throwaway schema and let the server normalise both sides for us.
_MIRROR_SCHEMA = 'declared'
# NOTE: cast contype because asyncpg hands PostgreSQL's "char" back as bytes
_CONSTRAINTS = """
select n.nspname, c.conrelid::regclass::text, c.conname, c.contype::text,
       pg_get_constraintdef(c.oid)
from pg_constraint c join pg_namespace n on n.oid = c.connamespace
where n.nspname = any(:schemas) and c.contype in ('c', 'f', 'p', 'u')
"""
# Everything a migration can leave behind, not just tables. Naming no object and no
# kind is the point: an enum, a sequence, a view or a function that upgrade creates and
# downgrade forgets shows up as an addition to this set whatever it is called. Indexes
# and TOAST tables are omitted because they only ever exist as part of a table.
_OBJECTS = """
select 'relation:' || c.relkind::text, c.relname
from pg_class c join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public' and c.relkind in ('r', 'p', 'v', 'm', 'S', 'c', 'f')
union all
select 'type:' || t.typtype::text, t.typname
from pg_type t join pg_namespace n on n.oid = t.typnamespace
where n.nspname = 'public' and t.typtype in ('e', 'd')
union all
select 'routine', p.proname
from pg_proc p join pg_namespace n on n.oid = p.pronamespace
where n.nspname = 'public'
union all
select 'extension', extname from pg_extension
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


async def _public_objects(engine: AsyncEngine) -> set[tuple[str, str]]:
    async with engine.connect() as connection:
        result = await connection.execute(text(_OBJECTS))
        return {(row[0], row[1]) for row in result if row[1] != _VERSION_TABLE}


async def _public_tables(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as connection:
        result = await connection.execute(
            text("select tablename from pg_tables where schemaname = 'public'"),
        )
        return {row[0] for row in result}


def _build_mirror(connection: Connection) -> None:
    connection.execute(text(f'create schema {_MIRROR_SCHEMA}'))
    # NOTE: the convention has to come along or every unnamed constraint is renamed
    mirror = MetaData(
        schema=_MIRROR_SCHEMA,
        naming_convention=AppRecord.metadata.naming_convention,
    )
    for table in AppRecord.metadata.tables.values():
        table.to_metadata(mirror, schema=_MIRROR_SCHEMA)
    mirror.create_all(connection)


def _read_constraints(
    connection: Connection,
    tables: frozenset[str],
) -> dict[str, set[Constraint]]:
    rows = connection.execute(
        text(_CONSTRAINTS).bindparams(schemas=['public', _MIRROR_SCHEMA]),
    ).all()
    sides: dict[str, set[Constraint]] = {'public': set(), _MIRROR_SCHEMA: set()}
    for schema, relation, name, kind, definition in rows:
        table = relation.removeprefix(f'{_MIRROR_SCHEMA}.').strip('"')
        if table in tables:
            sides[schema].add(
                (table, name, kind, definition.replace(f'{_MIRROR_SCHEMA}.', '')),
            )
    return sides


@_integration
async def test_declared_constraint_definitions_match_the_migrated_schema(
    database_url: str,
) -> None:
    tables = mapped_tables()
    engine = create_async_engine(database_url)
    try:
        await _drive(engine, 'up', 'default@head')
        async with engine.begin() as connection:
            await connection.run_sync(_build_mirror)
        try:
            async with engine.connect() as connection:
                sides = await connection.run_sync(_read_constraints, tables)
        finally:
            async with engine.begin() as connection:
                await connection.execute(
                    text(f'drop schema {_MIRROR_SCHEMA} cascade'),
                )
    finally:
        await engine.dispose()

    migrated, declared = sides['public'], sides[_MIRROR_SCHEMA]

    assert declared, 'the mirror declared no constraint; the check would be vacuous'
    assert migrated - declared == set(), (
        'the migrated schema has constraints the ORM does not declare:\n'
        + '\n'.join(repr(entry) for entry in sorted(migrated - declared))
    )
    assert declared - migrated == set(), (
        'the ORM declares constraints the migrated schema does not have:\n'
        + '\n'.join(repr(entry) for entry in sorted(declared - migrated))
    )


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
        # NOTE: the database is shared across the suite and may already be at head, so
        # reach base first; only then does a snapshot mean "what upgrade did not create"
        await _drive(engine, 'down', 'default@base')
        baseline = await _public_objects(engine)

        await _drive(engine, 'up', 'default@head')
        assert tables <= await _public_tables(engine)

        await _drive(engine, 'down', 'default@base')
        remaining = await _public_tables(engine)
        leftovers = await _public_objects(engine) - baseline

        # NOTE: naming no table keeps this honest as schemas are added; the version
        # table is alembic's own and downgrading to base deliberately leaves it
        assert not (tables & remaining)
        assert remaining <= {_VERSION_TABLE}
        assert not leftovers, 'downgrade left objects behind:\n' + '\n'.join(
            f'{kind} {name}' for kind, name in sorted(leftovers)
        )

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
