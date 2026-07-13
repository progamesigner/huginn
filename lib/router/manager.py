from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import override

from fastapi import APIRouter, FastAPI
from starlette.middleware import Middleware
from starlette.types import ASGIApp

from .contracts import RouterContract

type Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


class RouterManager(RouterContract):
    application: FastAPI | None
    debug: bool
    docs: bool
    lifespan: Lifespan | None
    middlewares: tuple[Middleware, ...]
    routers: list[APIRouter]

    def __init__(
        self,
        *,
        debug: bool = False,
        docs: bool = False,
        lifespan: Lifespan | None = None,
        middlewares: Sequence[Middleware] = (),
    ) -> None:
        self.application = None
        self.debug = debug
        self.docs = docs
        self.lifespan = lifespan
        self.middlewares = tuple(middlewares)
        self.routers = []

    @property
    @override
    def app(self) -> ASGIApp:
        if self.application is None:
            self.application = FastAPI(
                debug=self.debug,
                docs_url='/docs' if self.docs else None,
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
            self.application.include_router(router)
