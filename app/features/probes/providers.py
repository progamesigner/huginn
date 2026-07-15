from typing import override

from app.core.providers.router import RouterProvider
from lib.container import ContainerContract, ProviderContract
from lib.router import RouterContract


class ProbesEndpointProvider(ProviderContract):
    dependencies = (RouterProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from .endpoints import router

        container.resolve(RouterContract).include(router)
