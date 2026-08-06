from typing import override

from app.core.config.registry import HuginnConfig
from app.features.agents.config import AgentConfig
from app.features.agents.contracts import ToolHarvest, ToolSource

from .binding import build_bound_tool
from .config import TOOLS
from .registry import ToolRegistry


class ConfiguredToolSource(ToolSource):
    config: HuginnConfig
    registry: ToolRegistry

    def __init__(self, config: HuginnConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    @override
    async def collect(self, agent: AgentConfig) -> ToolHarvest:
        configured = self.config.section(TOOLS).root
        return ToolHarvest(
            tools=tuple(
                (
                    ref.name,
                    build_bound_tool(
                        self.registry,
                        configured[ref.name],
                        ref.params,
                        ref.inject,
                    ),
                )
                for ref in agent.tools
            ),
        )
