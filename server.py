from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.types import ASGIApp
from uvicorn import run

from app.core.application import App
from app.core.providers import LoggingProvider, RouterProvider, SettingsProvider
from app.features.probes.providers import ProbesProvider
from lib.router import RouterContract


class ServerOptions(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='HUGINN_SERVER_')

    host: Annotated[str, Field(default='127.0.0.1')]
    port: Annotated[int, Field(default=8000)]
    reload: Annotated[bool, Field(default=False)]


def build() -> ASGIApp:
    container = App().build(
        SettingsProvider(),
        LoggingProvider(),
        RouterProvider(),
        ProbesProvider(),
    )
    return container.resolve(RouterContract).app


if __name__ == '__main__':
    options = ServerOptions()
    run(
        'server:build',
        factory=True,
        host=options.host,
        port=options.port,
        reload=options.reload,
        log_config=None,
    )
