from typing import override

from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.core.providers.temporal import ClockProvider
from app.features.mcp.providers import MCPProvider
from lib.container import ContainerContract, ProviderContract
from lib.mcp import MCPClientContract
from lib.observability import Logger
from lib.temporal import ClockContract

from .config import SKILLS, FilesystemSkillSourceConfig, MCPSkillSourceConfig
from .exceptions import SkillsNotConfiguredError
from .repositories import SkillRepository
from .sources.filesystem import FilesystemSkillSource
from .sources.mcp import MCPSkillSource


def _repository(container: ContainerContract) -> SkillRepository:
    settings = container.resolve(HuginnConfig).section(SKILLS)
    match settings.source:
        case MCPSkillSourceConfig() as config:
            source = MCPSkillSource(
                container.resolve(MCPClientContract),
                config.server,
                config.glob,
                list_tool=config.list_tool,
                read_tool=config.read_tool,
                logger=container.resolve(Logger),
            )
        case FilesystemSkillSourceConfig() as config:
            source = FilesystemSkillSource(
                config.root,
                config.glob,
                logger=container.resolve(Logger),
            )
        case None:
            raise SkillsNotConfiguredError('skills.source is not configured')

    return SkillRepository(source, container.resolve(ClockContract))


class SkillsProvider(ProviderContract):
    dependencies = (ClockProvider, ConfigProvider, MCPProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(SkillRepository, factory=_repository, shared=True)
