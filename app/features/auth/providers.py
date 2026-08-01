from typing import override

from app.contracts.auth import AuthContract
from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.providers.config import ConfigProvider
from app.features.agents.providers import AgentProvider
from lib.container import ContainerContract, ProviderContract

from .config import build_sections
from .services import ApiKeyDirectory


def _directory(container: ContainerContract) -> ApiKeyDirectory:
    return ApiKeyDirectory(container.resolve(HuginnConfig))


class AuthProvider(ProviderContract):
    dependencies = (AgentProvider, ConfigProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(AuthContract, factory=_directory, shared=True)

        sections = container.resolve(SectionRegistry)
        for spec in build_sections():
            sections.register(spec)
