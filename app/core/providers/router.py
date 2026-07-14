from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import override

from starlette.middleware import Middleware

from app.core.exceptions import AppError, handle_app_error, handle_unexpected_error
from app.core.middleware import RequestContextMiddleware
from app.core.settings import AppSettings
from lib.container import ContainerContract, ProviderContract
from lib.router import RouterContract, RouterManager

from .logging import LoggingProvider
from .settings import SettingsProvider


class RouterProvider(ProviderContract):
    dependencies = (SettingsProvider, LoggingProvider)

    def __init__(self, *, excludes: frozenset[str] = frozenset()) -> None:
        self.excludes = excludes

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
            exception_handlers={
                AppError: handle_app_error,
                Exception: handle_unexpected_error,
            },
            lifespan=lifespan,
            middlewares=[Middleware(RequestContextMiddleware, excludes=self.excludes)],
        )
        container.bind(RouterContract, instance=manager)
