from typing import override

from app.core.providers.logging import LoggingProvider
from app.core.providers.redis import RedisProvider
from app.core.providers.settings import SettingsProvider
from app.core.providers.temporal import ClockProvider
from lib.container import ContainerContract, ProviderContract


class BootstrapProvider(ProviderContract):
    dependencies = (SettingsProvider, LoggingProvider, RedisProvider, ClockProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
