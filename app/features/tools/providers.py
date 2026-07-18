from typing import override

from app.core.config import SectionRegistry
from app.core.providers.config import ConfigProvider
from app.core.providers.http import HTTPClientProvider
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract

from .config import build_sections
from .registry import ToolRegistry
from .websearch import WebSearchTool, WebSearchToolConfig, build_websearch


class ToolRegistryProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ToolRegistry, instance=ToolRegistry())


class ToolsConfigProvider(ProviderContract):
    dependencies = (ConfigProvider, ToolRegistryProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        sections = container.resolve(SectionRegistry)
        for spec in build_sections(container.resolve(ToolRegistry)):
            sections.register(spec)


class WebSearchToolProvider(ProviderContract):
    dependencies = (HTTPClientProvider, ToolRegistryProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        http = container.resolve(HTTPClientContract)
        container.resolve(ToolRegistry).register(
            WebSearchTool,
            WebSearchToolConfig,
            build=lambda config: build_websearch(http, config),
        )
