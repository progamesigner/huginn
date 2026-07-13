from typing import override

from app.core.settings import AppSettings
from lib.container import ContainerContract, ProviderContract


class SettingsProvider(ProviderContract):
    @override
    def register(self, container: ContainerContract) -> None:
        container.bind(AppSettings, instance=AppSettings())
