from collections.abc import Sequence
from typing import override

from app.core.config.registry import HuginnConfig
from app.features.agents.config import AgentConfig
from app.features.agents.contracts import NamedTool, ToolSource

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
    async def collect(self, agent: AgentConfig) -> Sequence[NamedTool]:
        configured = self.config.section(TOOLS).root
        return [
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
        ]
