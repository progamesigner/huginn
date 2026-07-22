import pytest

from app.core.application import App
from app.core.providers.redis import RedisProvider
from app.core.providers.settings import SettingsProvider
from lib.redis import RedisContract, RedisError, RedisManager
from tests.helpers.settings import make_settings


def test_provider_binds_shared_manager() -> None:
    settings = make_settings(redis_url='redis://cache:6379/2')
    container = App().build(SettingsProvider(settings), RedisProvider())
    manager = container.resolve(RedisContract)

    assert isinstance(manager, RedisManager)
    assert manager.urls == {None: 'redis://cache:6379/2'}
    assert container.resolve(RedisContract) is manager


def test_provider_defers_missing_url_until_resolution() -> None:
    container = App().build(
        SettingsProvider(make_settings()),
        RedisProvider(),
    )

    with pytest.raises(RedisError, match='not configured'):
        container.resolve(RedisContract)
