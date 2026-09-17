from typing import override

from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.providers.config import ConfigProvider
from app.core.providers.temporal import ClockProvider
from app.features.agents.contracts import ToolSourceRegistry
from app.features.agents.providers import AgentProvider
from app.features.mcp.providers import MCPProvider
from lib.container import ContainerContract, ProviderContract
from lib.mcp import MCPClientContract
from lib.temporal import ClockContract

from .config import SKILLS
from .exceptions import SkillsNotConfiguredError
from .repositories import SkillRepository
from .sources.muninn import MuninnSkillSource
from .tools import SkillToolSource


def _repository(container: ContainerContract) -> SkillRepository:
    settings = container.resolve(HuginnConfig).section(SKILLS)
    if settings.server is None:
        raise SkillsNotConfiguredError('skills.server is not configured')

    source = MuninnSkillSource(
        container.resolve(MCPClientContract),
        settings.server,
        settings.glob,
    )
    return SkillRepository(source, container.resolve(ClockContract))


class SkillsConfigProvider(ProviderContract):
    dependencies = (AgentProvider, ConfigProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.resolve(SectionRegistry).register(SKILLS)


class SkillsProvider(ProviderContract):
    dependencies = (SkillsConfigProvider, ClockProvider, MCPProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(SkillRepository, factory=_repository, shared=True)
        container.resolve(ToolSourceRegistry).register(
            lambda: SkillToolSource(
                lambda: container.resolve(SkillRepository),
                container.resolve(HuginnConfig).section(SKILLS),
            ),
        )
