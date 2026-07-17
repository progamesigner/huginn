from typing import override

from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract, HTTPClientManager


class HTTPClientProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(HTTPClientContract, instance=HTTPClientManager())
