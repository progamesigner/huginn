from typing import override

from app.contracts.channels import ChannelResolverContract
from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.core.providers.http import HTTPClientProvider
from app.features.conversations.providers import ConversationProvider
from app.features.identities.providers import IdentityProvider
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract

from .config import TRANSCRIPTION
from .contracts import TranscriptionContract
from .registry import ChannelKind, ChannelRegistry, Command, CommandRegistry
from .repositories import TranscriptionRepository
from .resolver import ChannelSecretResolver


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
        container.bind(
            ChannelRegistry,
            factory=lambda c: ChannelRegistry(c.collect(ChannelKind)),
            shared=True,
        )
        container.bind(
            CommandRegistry,
            factory=lambda c: CommandRegistry(c.collect(Command)),
            shared=True,
        )


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


class ChannelEndpointProvider(ProviderContract):
    dependencies = (ChannelProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
