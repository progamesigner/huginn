from typing import override

from app.features.agents.providers import AgentsConfigProvider
from app.features.tools.providers import WebSearchToolProvider
from lib.container import ContainerContract, ProviderContract


class FeaturesProvider(ProviderContract):
    dependencies = (AgentsConfigProvider, WebSearchToolProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
