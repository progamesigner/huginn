from typing import override

from app.exts.felo.client import FeloClient
from app.exts.felo.providers import FeloExtProvider
from lib.container import ContainerContract, ProviderContract

from .registry import ToolRegistry
from .websearch import WebSearchTool


class ToolRegistryProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ToolRegistry, instance=ToolRegistry())


class WebSearchToolProvider(ProviderContract):
    dependencies = (FeloExtProvider, ToolRegistryProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        registry = container.resolve(ToolRegistry)
        registry.register(
            WebSearchTool,
            factory=lambda: WebSearchTool(container.resolve(FeloClient)),
        )
