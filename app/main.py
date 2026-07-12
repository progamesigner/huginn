from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import ClassVar

from fastapi import FastAPI

from app.core.health import router as health_router
from app.core.providers import SettingsProvider
from app.core.settings import AppSettings
from lib.container import Container, ProviderContract


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with app.state.container:
        yield


class App(Container):
    providers: ClassVar[tuple[ProviderContract, ...]] = (SettingsProvider(),)


def create() -> FastAPI:
    container = App().build()
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.container = container
    app.state.settings = container.resolve(AppSettings)
    app.include_router(health_router)
    return app
