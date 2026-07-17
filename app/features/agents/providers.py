from typing import override

from app.core.config import SectionRegistry
from app.core.providers.config import ConfigProvider
from app.exts.felo.client import FeloClient
from app.exts.felo.providers import FeloExtProvider
from lib.container import ContainerContract, ProviderContract

from .config import build_sections
from .tools import ToolRegistry, WebSearchTool


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


class AgentsConfigProvider(ProviderContract):
    dependencies = (ConfigProvider, ToolRegistryProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        sections = container.resolve(SectionRegistry)
        for spec in build_sections(container.resolve(ToolRegistry)):
            sections.register(spec)
