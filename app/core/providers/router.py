from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import override

from starlette.middleware import Middleware

from app.core.exceptions import AppError
from app.core.middleware import RequestContextMiddleware
from app.core.renderers import render_app_error, render_unexpected_error
from app.core.settings import HuginnSettings
from lib.container import ContainerContract, ProviderContract
from lib.router import RouterContract, RouterManager

from .logging import LoggingProvider
from .settings import SettingsProvider


class RouterProvider(ProviderContract):
    dependencies = (SettingsProvider, LoggingProvider)

    def __init__(self, *, excludes: frozenset[str] = frozenset()) -> None:
        self.excludes = excludes

    @override
    def __call__(self, container: ContainerContract) -> None:
        develop = container.resolve(HuginnSettings).running_in_mode('dev')

        @asynccontextmanager
        async def lifespan(_: object) -> AsyncIterator[None]:
            async with container:
                yield

        manager = RouterManager(
            debug=develop,
            docs=develop,
            exception_handlers={
                AppError: render_app_error,
                Exception: render_unexpected_error,
            },
            lifespan=lifespan,
            middlewares=[Middleware(RequestContextMiddleware, excludes=self.excludes)],
        )
        container.bind(RouterContract, instance=manager)
