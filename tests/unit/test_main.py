from typing import cast

import httpx
import pytest
from starlette.applications import Starlette

from app.core.application import App
from app.core.providers import LoggingProvider, RouterProvider, SettingsProvider
from app.main import create
from lib.router import RouterContract, RouterManager


def test_create_resolves_router_manager() -> None:
    container = App().build(
        SettingsProvider(),
        LoggingProvider(),
        RouterProvider(),
    )

    assert isinstance(container.resolve(RouterContract), RouterManager)


async def test_dev_environment_exposes_docs() -> None:
    transport = httpx.ASGITransport(app=create())
    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://huginn',
    ) as client:
        response = await client.get('/docs')

    assert response.status_code == 200


async def test_prod_environment_hides_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HUGINN_ENVIRONMENT', 'prod')

    transport = httpx.ASGITransport(app=create())
    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://huginn',
    ) as client:
        response = await client.get('/docs')

    assert response.status_code == 404


async def test_lifespan_runs_container_through_probes() -> None:
    app = cast(Starlette, create())

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url='http://huginn',
        ) as client:
            response = await client.get('/healthz')

    assert response.status_code == 200
