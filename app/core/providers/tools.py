from typing import override

from app.core.tools import ToolRegistry
from lib.container import ContainerContract, ProviderContract


class ToolRegistryProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ToolRegistry, instance=ToolRegistry())
