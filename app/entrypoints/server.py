from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.types import ASGIApp
from uvicorn import run

from app.core.application import App
from app.core.providers.router import RouterProvider
from app.features.probes.providers import ProbesEndpointProvider
from lib.router import RouterContract


class ServerOptions(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='HUGINN_SERVER_')

    host: Annotated[str, Field(default='127.0.0.1')]
    port: Annotated[int, Field(default=8000)]
    reload: Annotated[bool, Field(default=False)]


def create() -> ASGIApp:
    container = App().build(
        *App.providers,
        RouterProvider(excludes=frozenset({'/healthz', '/readyz'})),
        ProbesEndpointProvider(),
    )
    return container.resolve(RouterContract).app


def main() -> None:
    options = ServerOptions()
    run(
        'app.entrypoints.server:create',
        factory=True,
        host=options.host,
        port=options.port,
        reload=options.reload,
        log_config=None,
    )


if __name__ == '__main__':
    main()
