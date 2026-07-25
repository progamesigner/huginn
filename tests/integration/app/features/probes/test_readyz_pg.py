from unittest.mock import AsyncMock, MagicMock

import pytest

from app.features.probes.endpoints import readyz
from lib.database import DatabaseManager
from lib.mcp import MCPClientContract
from lib.redis import RedisContract

pytestmark = pytest.mark.integration

_UNREACHABLE = 'postgresql+asyncpg://huginn:huginn@127.0.0.1:1/huginn'


def _mcp() -> MagicMock:
    mcp = MagicMock(spec=MCPClientContract)
    mcp.unavailable.return_value = ()
    return mcp


def _redis() -> MagicMock:
    client = MagicMock()
    client.ping = AsyncMock()
    connection = MagicMock()
    connection.__aenter__ = AsyncMock(return_value=client)
    connection.__aexit__ = AsyncMock(return_value=False)

    redis = MagicMock(spec=RedisContract)
    redis.connect.return_value = connection
    return redis


async def test_readyz_reaches_a_real_database(database_url: str) -> None:
    database = DatabaseManager({None: database_url})

    try:
        response = await readyz(database, _mcp(), _redis())
    finally:
        await database.dispose()

    assert response == {'status': 'ok', 'mcp': 'ok'}


async def test_readyz_fails_when_the_database_refuses_connections() -> None:
    database = DatabaseManager({None: _UNREACHABLE})

    try:
        with pytest.raises(ConnectionRefusedError):
            await readyz(database, _mcp(), _redis())
    finally:
        await database.dispose()
