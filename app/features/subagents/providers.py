from typing import override

from app.features.agents.providers import AgentProvider
from app.features.assets.providers import AssetsProvider
from lib.container import ContainerContract, ProviderContract


class SubagentsProvider(ProviderContract):
    dependencies = (AgentProvider, AssetsProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
