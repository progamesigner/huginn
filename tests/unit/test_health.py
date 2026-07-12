from collections.abc import AsyncIterator

import httpx
import pytest

from app.main import create


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create())
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
