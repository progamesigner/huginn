from typing import override

from app.core.providers.http import HTTPClientProvider
from app.features.agents.providers import AgentProvider
from app.features.blobs.providers import BlobProvider
from lib.container import ContainerContract, ProviderContract

from .registry import ToolKind, ToolRegistry
from .websearch.repositories import WebSearchRepository


class ToolRegistryProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            ToolRegistry,
            factory=lambda c: ToolRegistry(c.collect(ToolKind)),
            shared=True,
        )


class ToolProvider(ProviderContract):
    dependencies = (
        AgentProvider,
        BlobProvider,
        HTTPClientProvider,
        ToolRegistryProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            WebSearchRepository,
            constructor=WebSearchRepository,
            shared=True,
        )
