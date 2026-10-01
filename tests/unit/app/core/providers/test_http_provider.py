from app.core.providers.http import HTTPClientProvider
from lib.http import HTTPClientContract, HTTPClientManager
from tests.helpers.settings import app_with, make_settings


def test_provider_binds_shared_manager() -> None:
    container = app_with(make_settings()).build(HTTPClientProvider())
    manager = container.resolve(HTTPClientContract)

    assert isinstance(manager, HTTPClientManager)
    assert container.resolve(HTTPClientContract) is manager


def test_provider_applies_the_configured_keepalive_expiry() -> None:
    settings = make_settings(http_keepalive_expiry=90.0)
    container = app_with(settings).build(HTTPClientProvider())
    manager = container.resolve(HTTPClientContract)

    assert isinstance(manager, HTTPClientManager)
    assert manager.limits.keepalive_expiry == 90.0


def test_provider_defaults_the_keepalive_expiry_to_five_minutes() -> None:
    container = app_with(make_settings()).build(HTTPClientProvider())
    manager = container.resolve(HTTPClientContract)

    assert isinstance(manager, HTTPClientManager)
    assert manager.limits.keepalive_expiry == 300.0
