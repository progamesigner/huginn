from typing import override

from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.providers.config import ConfigProvider
from app.core.providers.http import HTTPClientProvider
from app.features.conversations.contracts import ConversationControlContract
from app.features.conversations.providers import ConversationProvider
from app.features.identities.contracts import (
    IdentityLinkingContract,
    IdentityResolverContract,
)
from app.features.identities.providers import IdentityProvider
from app.ports.channels import ChannelResolverContract
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract

from .config import TRANSCRIPTION, build_sections
from .contracts import TranscriptionContract
from .handlers import CommandHandlers
from .registry import ChannelRegistry, CommandRegistry
from .repositories import TranscriptionRepository
from .resolver import ChannelSecretResolver
from .telegram.config import TelegramChannelConfig
from .telegram.providers import TelegramEndpointProvider


def _resolver(container: ContainerContract) -> ChannelSecretResolver:
    return ChannelSecretResolver(container.resolve(HuginnConfig))


def _commands(container: ContainerContract) -> None:
    registry = container.resolve(CommandRegistry)
    handlers = CommandHandlers(
        lambda: container.resolve(IdentityResolverContract),
        lambda: container.resolve(IdentityLinkingContract),
        lambda: container.resolve(ConversationControlContract),
        registry,
    )
    for name, description, handler in handlers.commands():
        registry.register(name, description, handler)


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
    dependencies = (
        ChannelRegistryProvider,
        ConfigProvider,
        ConversationProvider,
        HTTPClientProvider,
        IdentityProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ChannelResolverContract, factory=_resolver)
        container.bind(TranscriptionContract, factory=_transcription, shared=True)
        _commands(container)

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
