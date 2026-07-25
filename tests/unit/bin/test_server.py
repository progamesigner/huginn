from typing import cast

import httpx
import pytest
from starlette.applications import Starlette
from starlette.types import ASGIApp

from app.core.application import App
from app.core.config.registry import SectionRegistry
from app.core.providers.bootstrap import BootstrapProvider
from app.core.providers.logging import LoggingProvider
from app.core.providers.router import RouterProvider
from app.core.providers.settings import SettingsProvider
from app.core.settings import HuginnSettings
from app.features.probes.providers import ProbesEndpointProvider
from app.features.providers import FeaturesProvider
from app.features.tools.registry import ToolRegistry
from bin.server import ServerOptions, build, create
from lib.router import RouterContract, RouterManager
from tests.helpers.settings import make_settings


def test_create_resolves_router_manager() -> None:
    container = App().build(
        SettingsProvider(),
        LoggingProvider(),
        RouterProvider(),
    )

    assert isinstance(container.resolve(RouterContract), RouterManager)


def create_with_settings(settings: HuginnSettings) -> ASGIApp:
    return (
        App()
        .build(
            SettingsProvider(settings),
            BootstrapProvider(),
            FeaturesProvider(),
            RouterProvider(excludes=frozenset({'/healthz', '/readyz'})),
            ProbesEndpointProvider(),
        )
        .resolve(RouterContract)
        .app
    )


def test_build_wires_features() -> None:
    container = build()

    assert 'websearch' in container.resolve(ToolRegistry).tools
    section_keys = {spec.key for spec in container.resolve(SectionRegistry)}
    assert {'agents', 'tools'} <= section_keys


async def test_server_boots_without_config_file() -> None:
    settings = make_settings(config_file='does-not-exist.yaml')
    transport = httpx.ASGITransport(app=create_with_settings(settings))
    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://huginn',
    ) as client:
        response = await client.get('/healthz')

    assert response.status_code == 200


async def test_dev_environment_exposes_docs() -> None:
    transport = httpx.ASGITransport(app=create())
    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://huginn',
    ) as client:
        response = await client.get('/docs')

    assert response.status_code == 200


async def test_prod_environment_hides_docs() -> None:
    transport = httpx.ASGITransport(
        app=create_with_settings(make_settings(environment='prod'))
    )
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
