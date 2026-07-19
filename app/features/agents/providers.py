from typing import override

from app.core.config.registry import SectionRegistry
from app.core.providers.config import ConfigProvider
from app.features.tools.providers import McpConfigProvider, ToolsConfigProvider
from lib.container import ContainerContract, ProviderContract

from .config import build_sections


class AgentsConfigProvider(ProviderContract):
    dependencies = (ConfigProvider, ToolsConfigProvider, McpConfigProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        sections = container.resolve(SectionRegistry)
        for spec in build_sections():
            sections.register(spec)
