from typing import override

from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.core.providers.temporal import ClockProvider
from app.features.assets.config import (
    AssetGlobs,
    AssetsConfig,
    FilesystemAssetSourceConfig,
    MCPAssetSourceConfig,
    asset_settings,
)
from app.features.assets.contracts import AssetsServiceContract
from app.features.assets.providers import AssetsProvider, build_assets
from app.features.mcp.providers import MCPProvider
from lib.container import ContainerContract, ProviderContract
from lib.mcp import MCPClientContract
from lib.observability import Logger
from lib.temporal import ClockContract

from .config import FilesystemSkillSourceConfig, MCPSkillSourceConfig, effective_skills
from .repositories import SkillRepositories, SkillRepository
from .sources.assets import AssetSkillSource


def _repository(container: ContainerContract, name: str) -> SkillRepository:
    configured = container.resolve(HuginnConfig)
    settings = effective_skills(configured)
    if name in asset_settings(configured).sources:
        return SkillRepository(
            AssetSkillSource(container.resolve(AssetsServiceContract), name),
            container.resolve(ClockContract),
        )
    match settings.resolve_source(name):
        case MCPSkillSourceConfig() as config:
            source = MCPAssetSourceConfig(
                server=config.server,
                globs=AssetGlobs(skill=config.glob),
                list_tool=config.list_tool,
                read_tool=config.read_tool,
                inject=config.inject,
                bind=config.bind,
            )
        case FilesystemSkillSourceConfig() as config:
            source = FilesystemAssetSourceConfig(
                kind='filesystem',
                root=config.root,
                globs=AssetGlobs(skill=config.glob),
            )
    assets = build_assets(
        AssetsConfig(sources={name: source}),
        container.resolve(MCPClientContract)
        if isinstance(source, MCPAssetSourceConfig)
        else None,
        container.resolve(ClockContract),
        logger=container.resolve(Logger),
    )
    return SkillRepository(
        AssetSkillSource(assets, name),
        container.resolve(ClockContract),
    )


class SkillsProvider(ProviderContract):
    dependencies = (AssetsProvider, ClockProvider, ConfigProvider, MCPProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            SkillRepositories,
            factory=lambda c: SkillRepositories(lambda name: _repository(c, name)),
            shared=True,
        )
