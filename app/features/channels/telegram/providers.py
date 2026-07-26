from typing import override

from app.core.channels.contracts import ChannelMessageHandlerContract
from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.core.providers.http import HTTPClientProvider
from app.core.providers.redis import RedisProvider
from app.core.providers.router import RouterProvider
from app.core.providers.temporal import ClockProvider
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract
from lib.redis import RedisContract
from lib.router import RouterContract
from lib.temporal import ClockContract

from .actor import TelegramChatActor, log_turn_batch
from .service import TelegramMessageHandler
from .tasks import BackgroundTasks


def _actor(container: ContainerContract) -> TelegramChatActor:
    return TelegramChatActor(container.resolve(RedisContract), log_turn_batch)


def _handler(container: ContainerContract) -> TelegramMessageHandler:
    return TelegramMessageHandler(
        config=container.resolve(HuginnConfig),
        redis=container.resolve(RedisContract),
        http=container.resolve(HTTPClientContract),
        clock=container.resolve(ClockContract),
        actor=container.resolve(TelegramChatActor),
        background=container.resolve(BackgroundTasks),
    )


class TelegramProvider(ProviderContract):
    dependencies = (ClockProvider, ConfigProvider, HTTPClientProvider, RedisProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(BackgroundTasks, instance=BackgroundTasks())
        container.bind(TelegramChatActor, factory=_actor)
        container.bind(TelegramMessageHandler, factory=_handler)
        container.bind(ChannelMessageHandlerContract, factory=_handler)


class TelegramEndpointProvider(ProviderContract):
    dependencies = (RouterProvider, TelegramProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from app.exts.telegram.endpoints import router

        container.resolve(RouterContract).include(router)
