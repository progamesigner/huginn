from app.core.application import App
from app.core.providers.http import HTTPClientProvider
from app.core.providers.settings import SettingsProvider
from lib.http import HTTPClientContract, HTTPClientManager
from tests.helpers.settings import make_settings


def test_provider_binds_shared_manager() -> None:
    container = App().build(SettingsProvider(make_settings()), HTTPClientProvider())
    manager = container.resolve(HTTPClientContract)

    assert isinstance(manager, HTTPClientManager)
    assert container.resolve(HTTPClientContract) is manager


def test_provider_applies_the_configured_keepalive_expiry() -> None:
    settings = make_settings(http_keepalive_expiry=90.0)
    container = App().build(SettingsProvider(settings), HTTPClientProvider())
    manager = container.resolve(HTTPClientContract)

    assert isinstance(manager, HTTPClientManager)
    assert manager.limits.keepalive_expiry == 90.0


def test_provider_defaults_the_keepalive_expiry_to_five_minutes() -> None:
    container = App().build(SettingsProvider(make_settings()), HTTPClientProvider())
    manager = container.resolve(HTTPClientContract)

    assert isinstance(manager, HTTPClientManager)
    assert manager.limits.keepalive_expiry == 300.0
