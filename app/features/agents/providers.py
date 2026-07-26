from typing import override

from app.core.config.registry import SectionRegistry
from app.core.providers.config import ConfigProvider
from app.features.tools.providers import MCPConfigProvider, ToolsConfigProvider
from app.features.tools.registry import ToolRegistry
from lib.container import ContainerContract, ProviderContract

from .config import build_sections


class AgentProvider(ProviderContract):
    dependencies = (ConfigProvider, ToolsConfigProvider, MCPConfigProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        sections = container.resolve(SectionRegistry)
        for spec in build_sections(container.resolve(ToolRegistry)):
            sections.register(spec)
