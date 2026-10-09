from typing import override

from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.core.providers.http import HTTPClientProvider
from app.features.agents.providers import AgentProvider
from app.features.blobs.config import BLOBS
from app.features.blobs.contracts import BlobServiceContract
from app.features.blobs.providers import BlobProvider
from lib.container import ContainerContract, ProviderContract

from .artifacts import ArtifactWriter
from .images.repositories import ImageRepository
from .registry import ToolKind, ToolRegistry
from .speech.repositories import SpeechRepository
from .websearch.repositories import WebSearchRepository


def _writer(container: ContainerContract) -> ArtifactWriter:
    return ArtifactWriter(
        container.resolve(BlobServiceContract),
        container.resolve(HuginnConfig).section(BLOBS).retention.generated,
    )


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
        ConfigProvider,
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
        container.bind(ImageRepository, constructor=ImageRepository, shared=True)
        container.bind(SpeechRepository, constructor=SpeechRepository, shared=True)
        container.bind(ArtifactWriter, factory=_writer, shared=True)
