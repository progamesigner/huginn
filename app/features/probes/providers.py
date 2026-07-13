from typing import override

from app.core.providers import RouterProvider
from lib.container import ContainerContract, ProviderContract
from lib.router import RouterContract


class ProbesProvider(ProviderContract):
    dependencies = (RouterProvider,)

    @override
    def register(self, container: ContainerContract) -> None:
        from .endpoints import router

        container.resolve(RouterContract).include(router)
