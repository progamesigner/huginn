from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import override

from starlette.middleware import Middleware

from app.core.middleware import RequestContextMiddleware
from app.core.settings import AppSettings
from lib.container import ContainerContract, ProviderContract
from lib.router import RouterContract, RouterManager

from .logging import LoggingProvider
from .settings import SettingsProvider


class RouterProvider(ProviderContract):
    dependencies = (SettingsProvider, LoggingProvider)

    @override
    def register(self, container: ContainerContract) -> None:
        develop = container.resolve(AppSettings).running_in_mode('dev')

        @asynccontextmanager
        async def lifespan(_: object) -> AsyncIterator[None]:
            async with container:
                yield

        manager = RouterManager(
            debug=develop,
            docs=develop,
            lifespan=lifespan,
            middlewares=[
                Middleware(
                    RequestContextMiddleware,
                    excludes=frozenset({'/healthz', '/readyz'}),
                )
            ],
        )
        container.bind(RouterContract, instance=manager)
