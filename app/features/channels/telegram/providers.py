from typing import override

from app.core.config.registry import HuginnConfig
from app.core.providers.router import RouterProvider
from lib.container import ContainerContract, ProviderContract
from lib.redis import RedisContract
from lib.router import RouterContract

from .handlers import LoggingUpdateHandler, TelegramUpdateHandler
from .service import TelegramWebhookService
from .tasks import BackgroundTasks


class TelegramWebhookEndpointProvider(ProviderContract):
    dependencies = (RouterProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from .endpoints import router

        def _service(container: ContainerContract) -> TelegramWebhookService:
            return TelegramWebhookService(
                config=container.resolve(HuginnConfig),
                redis=container.resolve(RedisContract),
                handler=container.resolve(TelegramUpdateHandler),
                background=container.resolve(BackgroundTasks),
            )

        container.bind(BackgroundTasks, instance=BackgroundTasks())
        container.bind(TelegramUpdateHandler, instance=LoggingUpdateHandler())
        container.bind(TelegramWebhookService, factory=_service)

        container.resolve(RouterContract).include(router)
