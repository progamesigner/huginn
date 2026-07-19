from collections.abc import Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, override

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.routing import APIRoute
from starlette.middleware import Middleware
from starlette.types import ASGIApp

from .contracts import RouterContract
from .exceptions import DuplicateRouteError

type ExceptionHandlers = dict[
    int | type[Exception],
    Callable[[Request, Any], Coroutine[Any, Any, Response]],
]
type Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def _reject_duplicate_routes(router: APIRouter, seen: set[tuple[str, str]]) -> None:
    for route in router.routes:
        if not isinstance(route, APIRoute):
            continue

        for method in route.methods or ():
            if (route.path, method) in seen:
                raise DuplicateRouteError(f'duplicate route: {method} {route.path}')

            seen.add((route.path, method))


class RouterManager(RouterContract):
    application: FastAPI | None
    debug: bool
    docs: bool
    exception_handlers: ExceptionHandlers
    lifespan: Lifespan | None
    middlewares: tuple[Middleware, ...]
    routers: list[APIRouter]
    seen_routes: set[tuple[str, str]]

    def __init__(
        self,
        *,
        debug: bool = False,
        docs: bool = False,
        exception_handlers: ExceptionHandlers | None = None,
        lifespan: Lifespan | None = None,
        middlewares: Sequence[Middleware] = (),
    ) -> None:
        self.application = None
        self.debug = debug
        self.docs = docs
        self.exception_handlers = dict(exception_handlers) if exception_handlers else {}
        self.lifespan = lifespan
        self.middlewares = tuple(middlewares)
        self.routers = []
        self.seen_routes = set()

    @property
    @override
    def app(self) -> ASGIApp:
        if self.application is None:
            for router in self.routers:
                _reject_duplicate_routes(router, self.seen_routes)

            self.application = FastAPI(
                debug=self.debug,
                docs_url='/docs' if self.docs else None,
                exception_handlers=self.exception_handlers,
                lifespan=self.lifespan,
                middleware=self.middlewares,
                openapi_url='/openapi.json' if self.docs else None,
                redoc_url='/redoc' if self.docs else None,
            )

            for router in self.routers:
                self.application.include_router(router)

        return self.application

    @override
    def include(self, router: APIRouter) -> None:
        if self.application is None:
            self.routers.append(router)
        else:
            _reject_duplicate_routes(router, self.seen_routes)
            self.application.include_router(router)
