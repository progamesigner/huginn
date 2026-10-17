from typing import override

from app.core.config.registry import HuginnConfig
from app.core.providers.guid import GUIDProvider
from app.core.providers.redis import RedisProvider
from app.core.providers.router import RouterProvider
from app.core.providers.temporal import ClockProvider
from app.features.agents.contracts import TurnServiceContract
from app.features.agents.providers import TurnProvider
from app.features.auth.providers import AuthProvider
from app.features.blobs.config import BLOBS
from app.features.blobs.contracts import BlobServiceContract
from app.features.blobs.providers import BlobProvider
from app.features.channels.contracts import AttachmentReadingServiceContract
from app.features.channels.providers import ChannelProvider
from app.features.identities.contracts import IdentityResolutionServiceContract
from app.features.identities.providers import IdentityProvider
from lib.container import ContainerContract, ProviderContract
from lib.guid import GUIDContract
from lib.observability import Logger
from lib.redis import RedisContract
from lib.router import RouterContract
from lib.temporal import ClockContract

from .attachments import AttachmentIntake
from .handlers import ChatHandlers
from .services import ChatService


def _handlers(container: ContainerContract) -> ChatHandlers:
    return ChatHandlers(
        container.resolve(ChatService),
        container.resolve(GUIDContract),
        container.resolve(ClockContract),
    )


def _service(container: ContainerContract) -> ChatService:
    return ChatService(
        container.resolve(TurnServiceContract),
        container.resolve(RedisContract),
        container.resolve(GUIDContract),
        container.resolve(BlobServiceContract),
        AttachmentIntake(
            container.resolve(BlobServiceContract),
            container.resolve(AttachmentReadingServiceContract),
        ),
        identities=lambda: container.resolve(IdentityResolutionServiceContract),
        uploads=container.resolve(HuginnConfig).section(BLOBS).retention.generated,
        logger=container.resolve(Logger),
    )


class MessageProvider(ProviderContract):
    dependencies = (
        AuthProvider,
        IdentityProvider,
        BlobProvider,
        ChannelProvider,
        ClockProvider,
        GUIDProvider,
        RedisProvider,
        TurnProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ChatService, factory=_service, shared=True)
        container.bind(ChatHandlers, factory=_handlers, shared=True)


class MessageEndpointProvider(ProviderContract):
    dependencies = (MessageProvider, RouterProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from .endpoints import router

        container.resolve(RouterContract).include(router)
