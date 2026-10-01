from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.types import ASGIApp
from uvicorn import run

from app.core.application import App
from app.core.providers.bootstrap import BootstrapProvider
from app.core.providers.settings import SettingsProvider
from app.providers import HuginnEndpointProvider, HuginnProvider
from lib.router import RouterContract


class ServerOptions(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='HUGINN_SERVER_')

    host: Annotated[str, Field()] = '127.0.0.1'
    port: Annotated[int, Field()] = 8000
    reload: Annotated[bool, Field()] = False
    timeout_graceful_shutdown: Annotated[int, Field()] = 60


def build() -> App:
    return App().build(
        SettingsProvider(),
        BootstrapProvider(),
        HuginnProvider(),
        HuginnEndpointProvider(),
    )


def create() -> ASGIApp:
    return build().resolve(RouterContract).app


def main() -> None:
    options = ServerOptions()
    run(
        'bin.server:create',
        factory=True,
        host=options.host,
        port=options.port,
        reload=options.reload,
        timeout_graceful_shutdown=options.timeout_graceful_shutdown,
        log_config=None,
    )


if __name__ == '__main__':
    main()
