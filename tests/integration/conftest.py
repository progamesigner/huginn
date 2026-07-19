import os
from collections.abc import AsyncIterator, Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine


@pytest.fixture(scope='session')
def database_url() -> Iterator[str]:
    url = os.environ.get('HUGINN_DATABASE_URL')
    if not url:
        pytest.skip('HUGINN_DATABASE_URL is not set')

    yield url


@pytest.fixture(scope='session')
def _migrated(database_url: str) -> None:
    # NOTE: bring the shared test database to head once (idempotent)
    command.upgrade(Config(toml_file='pyproject.toml'), 'head')


@pytest.fixture
async def session(
    _migrated: None,
    database_url: str,
) -> AsyncIterator[AsyncSession]:
    # NOTE: each test runs inside one transaction rolled back on teardown -> isolation
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
