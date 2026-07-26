from typing import override

from app.core.config.registry import SectionRegistry
from app.core.providers.config import ConfigProvider
from lib.container import ContainerContract, ProviderContract

from .config import build_sections
from .contracts import ToolSourceRegistry


class AgentProvider(ProviderContract):
    dependencies = (ConfigProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ToolSourceRegistry, instance=ToolSourceRegistry())

        sections = container.resolve(SectionRegistry)
        for spec in build_sections():
            sections.register(spec)
