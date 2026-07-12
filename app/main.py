from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.health import router as health_router
from app.core.settings import AppSettings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield


def build_app() -> FastAPI:
    settings = AppSettings()
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.include_router(health_router)
    return app
