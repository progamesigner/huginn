from typing import override

from app.core.providers.logging import LoggingProvider
from app.core.providers.settings import SettingsProvider
from lib.container import ContainerContract, ProviderContract


class BootstrapProvider(ProviderContract):
    dependencies = (SettingsProvider, LoggingProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
