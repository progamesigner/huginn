from typing import override

from app.contracts.auth import AuthContract
from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.features.agents.providers import AgentProvider
from lib.container import ContainerContract, ProviderContract
from lib.observability import Logger

from .services import ApiKeyDirectory


def _directory(container: ContainerContract) -> ApiKeyDirectory:
    return ApiKeyDirectory(
        container.resolve(HuginnConfig),
        logger=container.resolve(Logger),
    )


class AuthProvider(ProviderContract):
    dependencies = (AgentProvider, ConfigProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(AuthContract, factory=_directory, shared=True)
