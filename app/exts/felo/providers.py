from typing import override

from app.core.providers.http import HTTPClientProvider
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientManager

from .client import FeloClient
from .settings import FeloExtSettings


def build_client(container: ContainerContract) -> FeloClient:
    settings = container.resolve(FeloExtSettings)
    return FeloClient(
        http=container.resolve(HTTPClientManager),
        api_key=settings.api_key,
        base_url=str(settings.base_url),
    )


def build_settings(container: ContainerContract) -> FeloExtSettings:
    return FeloExtSettings()


class FeloExtProvider(ProviderContract):
    dependencies = (HTTPClientProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(FeloClient, factory=build_client, shared=True, eager=True)
        container.bind(FeloExtSettings, factory=build_settings, shared=True)
