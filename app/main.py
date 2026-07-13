from starlette.types import ASGIApp

from app.core.application import App
from app.core.providers import LoggingProvider, RouterProvider, SettingsProvider
from lib.router import RouterContract


def create() -> ASGIApp:
    container = App().build(
        SettingsProvider(),
        LoggingProvider(),
        RouterProvider(),
    )
    return container.resolve(RouterContract).app
