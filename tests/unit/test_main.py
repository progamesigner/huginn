from typing import cast

import httpx
import pytest
from starlette.applications import Starlette

from app.core.application import App
from app.core.providers.logging import LoggingProvider
from app.core.providers.router import RouterProvider
from app.core.providers.settings import SettingsProvider
from lib.router import RouterContract, RouterManager
from server import ServerOptions, build


def test_create_resolves_router_manager() -> None:
    container = App().build(
        SettingsProvider(),
        LoggingProvider(),
        RouterProvider(),
    )

    assert isinstance(container.resolve(RouterContract), RouterManager)


async def test_dev_environment_exposes_docs() -> None:
    transport = httpx.ASGITransport(app=build())
    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://huginn',
    ) as client:
        response = await client.get('/docs')

    assert response.status_code == 200


async def test_prod_environment_hides_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HUGINN_ENVIRONMENT', 'prod')

    transport = httpx.ASGITransport(app=build())
    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://huginn',
    ) as client:
        response = await client.get('/docs')

    assert response.status_code == 404


async def test_lifespan_runs_container_through_probes() -> None:
    app = cast(Starlette, build())

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url='http://huginn',
        ) as client:
            response = await client.get('/healthz')

    assert response.status_code == 200


@pytest.fixture(autouse=True)
def clean_server_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('HUGINN_SERVER_HOST', 'HUGINN_SERVER_PORT', 'HUGINN_SERVER_RELOAD'):
        monkeypatch.delenv(name, raising=False)


def test_server_options_defaults() -> None:
    options = ServerOptions()

    assert options.host == '127.0.0.1'
    assert options.port == 8000
    assert options.reload is False


def test_server_options_read_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HUGINN_SERVER_HOST', 'localhost')
    monkeypatch.setenv('HUGINN_SERVER_PORT', '8001')
    monkeypatch.setenv('HUGINN_SERVER_RELOAD', 'true')

    options = ServerOptions()

    assert options.host == 'localhost'
    assert options.port == 8001
    assert options.reload is True


def test_server_options_ignore_unprefixed_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HUGINN_PORT', 'tcp://10.0.0.1:8000')

    options = ServerOptions()

    assert options.port == 8000
