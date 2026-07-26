from typing import override

from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.providers.config import ConfigProvider
from app.core.providers.temporal import ClockProvider
from app.features.mcp.providers import MCPProvider
from lib.container import ContainerContract, ProviderContract
from lib.mcp import MCPClientContract
from lib.temporal import ClockContract

from .config import SKILLS
from .exceptions import SkillsNotConfiguredError
from .repositories import SkillRepository
from .sources.muninn import MuninnSkillSource


def _repository(container: ContainerContract) -> SkillRepository:
    settings = container.resolve(HuginnConfig).section(SKILLS)
    if settings.server is None:
        raise SkillsNotConfiguredError('skills.server is not configured')

    source = MuninnSkillSource(
        container.resolve(MCPClientContract),
        settings.server,
        settings.inject,
        settings.glob,
    )
    return SkillRepository(source, container.resolve(ClockContract))


class SkillsConfigProvider(ProviderContract):
    dependencies = (ConfigProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.resolve(SectionRegistry).register(SKILLS)


class SkillsProvider(ProviderContract):
    dependencies = (SkillsConfigProvider, ClockProvider, MCPProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(SkillRepository, factory=_repository, shared=True)
