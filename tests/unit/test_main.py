from app.core.middleware import RequestContextMiddleware
from app.core.settings import AppSettings
from app.main import create


def test_create_binds_settings_in_container() -> None:
    app = create()

    settings = app.state.container.resolve(AppSettings)

    assert settings is app.state.settings


def test_create_wires_request_context_middleware() -> None:
    app = create()

    assert any(
        middleware.cls is RequestContextMiddleware for middleware in app.user_middleware
    )


async def test_lifespan_starts_and_closes_container() -> None:
    app = create()

    async with app.router.lifespan_context(app):
        assert app.state.container.resolve(AppSettings) is not None
