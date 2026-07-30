from typing import override

from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.providers.config import ConfigProvider
from app.core.providers.http import HTTPClientProvider
from app.ports.channels import ChannelResolverContract
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract

from .config import TRANSCRIPTION, build_sections
from .contracts import TranscriptionContract
from .registry import ChannelRegistry, CommandRegistry
from .repositories import TranscriptionRepository
from .resolver import ChannelSecretResolver
from .telegram.config import TelegramChannelConfig
from .telegram.providers import TelegramEndpointProvider


def _resolver(container: ContainerContract) -> ChannelSecretResolver:
    return ChannelSecretResolver(container.resolve(HuginnConfig))


def _transcription(container: ContainerContract) -> TranscriptionRepository:
    return TranscriptionRepository(
        container.resolve(HTTPClientContract),
        container.resolve(HuginnConfig).section(TRANSCRIPTION),
    )


class ChannelRegistryProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ChannelRegistry, instance=ChannelRegistry())
        container.bind(CommandRegistry, instance=CommandRegistry())


class ChannelProvider(ProviderContract):
    dependencies = (ChannelRegistryProvider, ConfigProvider, HTTPClientProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ChannelResolverContract, factory=_resolver)
        container.bind(TranscriptionContract, factory=_transcription, shared=True)

        registry = container.resolve(ChannelRegistry)
        registry.register('telegram', TelegramChannelConfig)

        sections = container.resolve(SectionRegistry)
        for spec in build_sections(registry):
            sections.register(spec)


class ChannelEndpointProvider(ProviderContract):
    dependencies = (ChannelProvider, TelegramEndpointProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
