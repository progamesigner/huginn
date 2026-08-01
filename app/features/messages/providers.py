from typing import override

from app.core.providers.guid import GUIDProvider
from app.core.providers.redis import RedisProvider
from app.core.providers.router import RouterProvider
from app.features.agents.contracts import TurnHandlerContract
from app.features.agents.providers import TurnProvider
from app.features.auth.providers import AuthProvider
from lib.container import ContainerContract, ProviderContract
from lib.guid import GUIDContract
from lib.redis import RedisContract
from lib.router import RouterContract

from .handlers import ChatHandlers
from .services import ChatService


def _handlers(container: ContainerContract) -> ChatHandlers:
    return ChatHandlers(container.resolve(ChatService))


def _service(container: ContainerContract) -> ChatService:
    return ChatService(
        container.resolve(TurnHandlerContract),
        container.resolve(RedisContract),
        container.resolve(GUIDContract),
    )


class MessageProvider(ProviderContract):
    dependencies = (AuthProvider, GUIDProvider, RedisProvider, TurnProvider)

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
