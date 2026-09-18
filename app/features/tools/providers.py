from typing import override

from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.providers.config import ConfigProvider
from app.core.providers.http import HTTPClientProvider
from app.features.agents.contracts import ToolSourceRegistry
from app.features.agents.providers import AgentProvider
from app.features.blobs.contracts import BlobStoreContract
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract

from .config import build_sections
from .document.config import DocumentToolConfig
from .document.tool import ReadDocumentTool, build_read_document
from .registry import ToolRegistry
from .sources import ConfiguredToolSource
from .websearch.config import WebSearchToolConfig
from .websearch.repositories import WebSearchRepository
from .websearch.tool import WebSearchTool, build_websearch


class ToolRegistryProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ToolRegistry, instance=ToolRegistry())


class ToolsConfigProvider(ProviderContract):
    dependencies = (AgentProvider, ConfigProvider, ToolRegistryProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        sections = container.resolve(SectionRegistry)
        for spec in build_sections(container.resolve(ToolRegistry)):
            sections.register(spec)


class ToolProvider(ProviderContract):
    dependencies = (HTTPClientProvider, ToolRegistryProvider, ToolsConfigProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        registry = container.resolve(ToolRegistry)
        http = container.resolve(HTTPClientContract)

        repository = WebSearchRepository(http)
        container.bind(WebSearchRepository, instance=repository)
        registry.register(
            WebSearchTool,
            WebSearchToolConfig,
            builder=lambda config, _scope: build_websearch(repository, config),
        )
        registry.register(
            ReadDocumentTool,
            DocumentToolConfig,
            builder=lambda config, scope: build_read_document(
                container.resolve(BlobStoreContract),
                config,
                scope,
            ),
        )

        container.resolve(ToolSourceRegistry).register(
            lambda: ConfiguredToolSource(
                container.resolve(HuginnConfig),
                registry,
            ),
        )
