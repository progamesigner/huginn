from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.application import App
from app.core.health import router as health_router
from app.core.providers import LoggingProvider, SettingsProvider
from app.core.settings import AppSettings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with app.state.container:
        yield


def create() -> FastAPI:
    container = App().build(
        SettingsProvider(),
        LoggingProvider(),
    )
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.container = container
    app.state.settings = container.resolve(AppSettings)
    app.include_router(health_router)
    return app
