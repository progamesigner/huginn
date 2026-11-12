from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from fastapi import FastAPI

from app.features.probes.endpoints import readyz
from bin.server import build
from lib.database import DatabaseContract
from lib.mcp import MCPClientContract
from lib.redis import RedisContract
from lib.router import RouterContract


def make_redis() -> MagicMock:
    client = MagicMock()
    client.ping = AsyncMock()
    connection = MagicMock()
    connection.__aenter__ = AsyncMock(return_value=client)
    connection.__aexit__ = AsyncMock(return_value=False)

    redis = MagicMock(spec=RedisContract)
    redis.connect.return_value = connection
    return redis


def make_database(*, failing: bool = False) -> MagicMock:
    database = MagicMock(spec=DatabaseContract)
    database.ping = AsyncMock(
        # NOTE: asyncpg lets the OSError through unwrapped; the integration test pins it
        side_effect=ConnectionRefusedError('Connect call failed') if failing else None,
    )
    return database


def make_mcp(*unavailable: str) -> MagicMock:
    mcp = MagicMock(spec=MCPClientContract)
    mcp.unavailable.return_value = unavailable
    return mcp


@pytest.fixture
async def client() -> AsyncIterator[httpx2.AsyncClient]:
    # NOTE: no config file in a unit test, so the eager MCP client dials nothing
    application = build()
    with (
        application.use(DatabaseContract, make_database()),
        application.use(RedisContract, make_redis()),
    ):
        app = application.resolve(RouterContract).app
        assert isinstance(app, FastAPI)
        async with app.router.lifespan_context(app):
            transport = httpx2.ASGITransport(app=app)
            async with httpx2.AsyncClient(
                transport=transport,
                base_url='http://huginn',
            ) as client:
                yield client


async def test_healthz(client: httpx2.AsyncClient) -> None:
    response = await client.get('/healthz')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


async def test_readyz(client: httpx2.AsyncClient) -> None:
    response = await client.get('/readyz')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'mcp': 'ok'}


async def test_readyz_pings_redis_and_the_database() -> None:
    database = make_database()
    redis = make_redis()

    response = await readyz(database, make_mcp(), redis)

    assert response == {'status': 'ok', 'mcp': 'ok'}
    redis.connect.return_value.__aenter__.return_value.ping.assert_awaited_once()
    database.ping.assert_awaited_once()


async def test_readyz_reports_unreachable_mcp_servers_without_failing() -> None:
    response = await readyz(make_database(), make_mcp('muninn', 'stock'), make_redis())

    assert response == {'status': 'ok', 'mcp': 'unavailable: muninn, stock'}


async def test_readyz_fails_when_the_database_is_unreachable() -> None:
    with pytest.raises(ConnectionRefusedError):
        await readyz(make_database(failing=True), make_mcp(), make_redis())
