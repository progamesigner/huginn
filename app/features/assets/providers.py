from typing import override

from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.core.providers.temporal import ClockProvider
from app.features.agents.providers import AgentProvider
from app.features.mcp.providers import MCPProvider
from lib.container import ContainerContract, ProviderContract
from lib.mcp import MCPClientContract
from lib.observability import Logger
from lib.temporal import ClockContract

from .config import AssetsConfig, MCPAssetSourceConfig, asset_settings
from .contracts import AssetsServiceContract
from .services import AssetsService


def build_assets(
    config: AssetsConfig,
    client: MCPClientContract | None,
    clock: ClockContract,
    *,
    logger: Logger,
) -> AssetsServiceContract:
    return AssetsService(config, client, clock, logger=logger)


class AssetsProvider(ProviderContract):
    dependencies = (AgentProvider, ClockProvider, ConfigProvider, MCPProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            AssetsServiceContract,
            factory=lambda c: AssetsService(
                asset_settings(c.resolve(HuginnConfig)),
                c.resolve(MCPClientContract)
                if any(
                    isinstance(source, MCPAssetSourceConfig)
                    for source in asset_settings(
                        c.resolve(HuginnConfig),
                    ).sources.values()
                )
                else None,
                c.resolve(ClockContract),
                logger=c.resolve(Logger),
            ),
            shared=True,
        )
