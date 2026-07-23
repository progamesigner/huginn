from typing import override

from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.providers.config import ConfigProvider
from app.core.providers.temporal import ClockProvider
from lib.container import ContainerContract, ProviderContract
from lib.mcp import MCPClientContract
from lib.temporal import ClockContract

from .config import SKILLS
from .repositories import SkillRepository
from .sources.muninn import MuninnSkillSource


class SkillsConfigProvider(ProviderContract):
    dependencies = (ConfigProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.resolve(SectionRegistry).register(SKILLS)


class SkillsProvider(ProviderContract):
    dependencies = (SkillsConfigProvider, ClockProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        settings = container.resolve(HuginnConfig).section(SKILLS)
        if settings.server is None:
            return

        source = MuninnSkillSource(
            container.resolve(MCPClientContract),
            settings.server,
            settings.inject,
            settings.glob,
        )
        container.bind(
            SkillRepository,
            instance=SkillRepository(source, container.resolve(ClockContract)),
        )
