from typing import override

from app.core.settings import HuginnSettings
from lib.container import ContainerContract, ProviderContract


class SettingsProvider(ProviderContract):
    def __init__(self, settings: HuginnSettings | None = None) -> None:
        self.settings = settings

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(HuginnSettings, instance=self.settings or HuginnSettings())
