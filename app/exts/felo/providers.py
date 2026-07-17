from typing import override

from app.core.providers.http import HTTPClientProvider
from app.core.providers.tools import ToolRegistryProvider
from app.core.tools import ToolRegistry
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientManager

from .client import FeloClient
from .settings import FeloExtSettings
from .tools import WebSearchTool


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


class FeloExtToolsProvider(ProviderContract):
    dependencies = (FeloExtProvider, ToolRegistryProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        registry = container.resolve(ToolRegistry)
        registry.register(
            WebSearchTool,
            factory=lambda: WebSearchTool(container.resolve(FeloClient)),
        )
