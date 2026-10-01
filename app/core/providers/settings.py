from typing import override

from app.core.settings import HuginnSettings
from lib.container import ContainerContract, ProviderContract


class SettingsProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        # NOTE: settings bound before build win, which is how tests pin their own
        if not container.bound(HuginnSettings):
            container.bind(HuginnSettings, instance=HuginnSettings())
