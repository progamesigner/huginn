from typing import override

from app.core.settings import HuginnSettings
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract, HTTPClientManager

from .settings import SettingsProvider


class HTTPClientProvider(ProviderContract):
    dependencies = (SettingsProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            HTTPClientContract,
            factory=lambda c: HTTPClientManager(
                keepalive_expiry=c.resolve(HuginnSettings).http_keepalive_expiry,
            ),
            shared=True,
        )
