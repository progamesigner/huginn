from collections.abc import AsyncIterator, Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.settings import HuginnSettings

_IMAGE = 'postgres:18'


@pytest.fixture(scope='session')
def database_url() -> Iterator[str]:
    configured = HuginnSettings().database_urls.get(None)
    if configured is not None:
        yield str(configured.get_secret_value())
        return

    try:
        from docker.errors import DockerException
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip('set HUGINN_DATABASE_URL or add the testcontainers dev dependency')

    try:
        container = PostgresContainer(_IMAGE, driver='asyncpg')
        with container:
            yield container.get_connection_url()
    except DockerException as error:
        pytest.skip(
            f'Docker unavailable; set HUGINN_DATABASE_URL to run db tests ({error})',
        )


@pytest.fixture(scope='session')
def _migrated(database_url: str) -> None:
    # NOTE: env.py resolves its dsn from settings, so expose the database there
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('HUGINN_DATABASE_URL', database_url)
        # NOTE: bring the shared test database to head once (idempotent)
        command.upgrade(Config(toml_file='pyproject.toml'), 'head')


@pytest.fixture
async def session(_migrated: None, database_url: str) -> AsyncIterator[AsyncSession]:
    # NOTE: each test runs inside one transaction rolled back on teardown
    engine = create_async_engine(database_url)
    connection = await engine.connect()
    transaction = await connection.begin()
    session = AsyncSession(bind=connection, join_transaction_mode='create_savepoint')

    try:
        yield session
    finally:
        await session.close()
        await transaction.rollback()
        await connection.close()
        await engine.dispose()
