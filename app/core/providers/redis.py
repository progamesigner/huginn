from typing import override

from app.core.settings import HuginnSettings
from lib.container import ContainerContract, ProviderContract
from lib.redis import RedisContract, RedisError, RedisManager

from .settings import SettingsProvider


class RedisProvider(ProviderContract):
    dependencies = (SettingsProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        def _factory(container: ContainerContract) -> RedisManager:
            configured = container.resolve(HuginnSettings).redis_urls
            if not configured:
                raise RedisError('redis urls are not configured')

            return RedisManager(
                {name: str(url.get_secret_value()) for name, url in configured.items()}
            )

        container.bind(RedisContract, factory=_factory, shared=True)
