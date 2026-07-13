from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast, override

from starlette.middleware import Middleware

from app.core.application import App
from app.core.health import router as health_router
from app.core.middleware import RequestContextMiddleware
from app.core.settings import AppSettings
from lib.container import ContainerContract, ProviderContract
from lib.observability import Logger, configure_logging, create_logger
from lib.router import RouterContract, RouterManager


class SettingsProvider(ProviderContract):
    @override
    def register(self, container: ContainerContract) -> None:
        container.bind(AppSettings, instance=AppSettings())


class LoggingProvider(ProviderContract):
    dependencies = (SettingsProvider,)

    @override
    def register(self, container: ContainerContract) -> None:
        configure_logging(
            develop=cast(App, container).running_in_mode('dev'),
            capture=('uvicorn', 'uvicorn.error'),
            silence=('uvicorn.access',),
        )
        container.bind(Logger, factory=lambda _: create_logger('huginn'))


class RouterProvider(ProviderContract):
    dependencies = (SettingsProvider, LoggingProvider)

    @override
    def register(self, container: ContainerContract) -> None:
        develop = cast(App, container).running_in_mode('dev')

        @asynccontextmanager
        async def lifespan(_: object) -> AsyncIterator[None]:
            async with container:
                yield

        manager = RouterManager(
            debug=develop,
            docs=develop,
            lifespan=lifespan,
            middlewares=[Middleware(RequestContextMiddleware)],
        )
        manager.include(health_router)
        container.bind(RouterContract, instance=manager)
