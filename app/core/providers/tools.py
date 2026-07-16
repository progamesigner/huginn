from typing import override

from app.core.tools import DEFAULT_PACKAGES, Packages, ToolRegistry, build_registry
from lib.container import ContainerContract, ProviderContract


class ToolRegistryProvider(ProviderContract):
    def __init__(self, *, packages: Packages = DEFAULT_PACKAGES) -> None:
        self.packages = packages

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ToolRegistry, instance=build_registry(self.packages))
