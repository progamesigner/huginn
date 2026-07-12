from app.core.settings import AppSettings
from app.main import App, create


def test_create_binds_settings_in_container() -> None:
    app = create()

    settings = app.state.container.resolve(AppSettings)

    assert settings is app.state.settings


def test_build_wires_default_providers() -> None:
    application = App().build()

    assert application.bound(AppSettings)


async def test_lifespan_starts_and_closes_container() -> None:
    app = create()

    async with app.router.lifespan_context(app):
        assert app.state.container.resolve(AppSettings) is not None
