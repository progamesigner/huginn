from typing import override

from lib.container import ContainerContract, ProviderContract
from lib.temporal import ClockContract, ClockManager


class ClockProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ClockContract, instance=ClockManager())
