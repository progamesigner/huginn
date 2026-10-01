from typing import override

from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.core.providers.temporal import ClockProvider
from app.features.mcp.providers import MCPProvider
from lib.container import ContainerContract, ProviderContract
from lib.mcp import MCPClientContract
from lib.observability import Logger
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
        settings.glob,
        logger=container.resolve(Logger),
    )
    return SkillRepository(source, container.resolve(ClockContract))


class SkillsProvider(ProviderContract):
    dependencies = (ClockProvider, ConfigProvider, MCPProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(SkillRepository, factory=_repository, shared=True)
