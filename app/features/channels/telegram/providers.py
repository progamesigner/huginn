from typing import override

from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.core.providers.http import HTTPClientProvider
from app.core.providers.redis import RedisProvider
from app.core.providers.router import RouterProvider
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract
from lib.redis import RedisContract
from lib.router import RouterContract

from .actor import TelegramChatActor, log_turn_batch
from .handlers import ReactionUpdateHandler, TelegramUpdateHandler
from .service import TelegramWebhookService
from .tasks import BackgroundTasks


def _actor(container: ContainerContract) -> TelegramChatActor:
    return TelegramChatActor(container.resolve(RedisContract), log_turn_batch)


def _service(container: ContainerContract) -> TelegramWebhookService:
    return TelegramWebhookService(
        config=container.resolve(HuginnConfig),
        redis=container.resolve(RedisContract),
        handler=container.resolve(TelegramUpdateHandler),
        actor=container.resolve(TelegramChatActor),
        background=container.resolve(BackgroundTasks),
    )


class TelegramProvider(ProviderContract):
    dependencies = (ConfigProvider, HTTPClientProvider, RedisProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        handler = ReactionUpdateHandler(container.resolve(HTTPClientContract))

        container.bind(BackgroundTasks, instance=BackgroundTasks())
        container.bind(TelegramUpdateHandler, instance=handler)
        container.bind(TelegramChatActor, factory=_actor)
        container.bind(TelegramWebhookService, factory=_service)


class TelegramEndpointProvider(ProviderContract):
    dependencies = (RouterProvider, TelegramProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from .endpoints import router

        container.resolve(RouterContract).include(router)
