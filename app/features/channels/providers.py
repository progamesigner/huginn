from typing import override

from app.core.config.registry import SectionRegistry
from app.core.providers.config import ConfigProvider
from lib.container import ContainerContract, ProviderContract

from .config import build_sections
from .registry import ChannelRegistry
from .telegram.config import TelegramChannelConfig


class ChannelRegistryProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ChannelRegistry, instance=ChannelRegistry())


class ChannelProvider(ProviderContract):
    dependencies = (ConfigProvider, ChannelRegistryProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        registry = container.resolve(ChannelRegistry)
        registry.register('telegram', TelegramChannelConfig)

        sections = container.resolve(SectionRegistry)
        for spec in build_sections(registry):
            sections.register(spec)
