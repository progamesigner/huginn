from typing import override

from app.contracts.channels import ChannelGatewayFactoryContract
from app.core.providers.http import HTTPClientProvider
from app.core.providers.router import RouterProvider
from app.core.providers.temporal import ClockProvider
from app.exts.telegram.gateways import TelegramGatewayFactory
from app.features.channels.telegram.providers import TelegramProvider
from lib.container import ContainerContract, ProviderContract
from lib.router import RouterContract

# NOTE: the vendor's transport meets the feature here, so the feature never names it


class TelegramGatewayProvider(ProviderContract):
    dependencies = (ClockProvider, HTTPClientProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            ChannelGatewayFactoryContract,
            constructor=TelegramGatewayFactory,
            shared=True,
        )


class TelegramEndpointProvider(ProviderContract):
    dependencies = (RouterProvider, TelegramGatewayProvider, TelegramProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from app.exts.telegram.endpoints import router

        container.resolve(RouterContract).include(router)
