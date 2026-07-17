from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.types import ASGIApp
from uvicorn import run

from app.core.application import App
from app.core.providers.router import RouterProvider
from app.features.probes.providers import ProbesEndpointProvider
from app.providers.bootstrap import BootstrapProvider
from lib.container import ProviderContract
from lib.router import RouterContract

ENDPOINTS: tuple[ProviderContract, ...] = (ProbesEndpointProvider(),)


class ServerOptions(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='HUGINN_SERVER_')

    host: Annotated[str, Field()] = '127.0.0.1'
    port: Annotated[int, Field()] = 8000
    reload: Annotated[bool, Field()] = False


def create() -> ASGIApp:
    container = App().build(
        BootstrapProvider(),
        RouterProvider(excludes=frozenset({'/healthz', '/readyz'})),
        *ENDPOINTS,
    )
    return container.resolve(RouterContract).app


def main() -> None:
    options = ServerOptions()
    run(
        'bin.server:create',
        factory=True,
        host=options.host,
        port=options.port,
        reload=options.reload,
        log_config=None,
    )


if __name__ == '__main__':
    main()
