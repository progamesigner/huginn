from typing import override

from lib.container import ContainerContract, ProviderContract
from lib.guid import GUIDContract, GUIDManager


class GUIDProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(GUIDContract, instance=GUIDManager())
