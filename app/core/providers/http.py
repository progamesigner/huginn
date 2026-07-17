from typing import override

from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientManager


class HTTPClientProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(HTTPClientManager, instance=HTTPClientManager())
