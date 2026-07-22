from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from app.features.probes.endpoints import readyz
from bin.server import build
from lib.redis import RedisContract
from lib.router import RouterContract


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    redis = MagicMock(spec=RedisContract)
    redis_client = MagicMock()
    redis_client.ping = AsyncMock()
    connection = MagicMock()
    connection.__aenter__ = AsyncMock(return_value=redis_client)
    connection.__aexit__ = AsyncMock(return_value=False)
    redis.connect.return_value = connection

    application = build()
    with application.use(RedisContract, redis):
        app = application.resolve(RouterContract).app
        assert isinstance(app, FastAPI)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url='http://huginn',
            ) as client:
                yield client


async def test_healthz(client: httpx.AsyncClient) -> None:
    response = await client.get('/healthz')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


async def test_readyz(client: httpx.AsyncClient) -> None:
    response = await client.get('/readyz')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


async def test_readyz_pings_redis() -> None:
    redis = MagicMock(spec=RedisContract)
    client = MagicMock()
    client.ping = AsyncMock()
    connection = MagicMock()
    connection.__aenter__ = AsyncMock(return_value=client)
    connection.__aexit__ = AsyncMock(return_value=False)
    redis.connect.return_value = connection

    response = await readyz(redis)

    assert response == {'status': 'ok'}
    client.ping.assert_awaited_once()
