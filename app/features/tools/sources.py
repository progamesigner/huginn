import logging
from typing import override

from app.core.config.registry import HuginnConfig
from app.features.agents.config import AgentConfig
from app.features.agents.contracts import ToolHarvest, ToolSource
from app.features.agents.types import TurnScope, resolve_bindings

from .binding import build_bound_tool
from .config import TOOLS
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ConfiguredToolSource(ToolSource):
    config: HuginnConfig
    registry: ToolRegistry

    def __init__(self, config: HuginnConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    @override
    async def collect(self, agent: AgentConfig, scope: TurnScope) -> ToolHarvest:
        configured = self.config.section(TOOLS).root
        tools = []
        for ref in agent.tools:
            inject = resolve_bindings(ref.inject, ref.bind, scope)
            if inject is None:
                logger.info(
                    'tool %r omitted because bound keys are unavailable: %s',
                    ref.name,
                    ', '.join(sorted(ref.bind)),
                )
                continue
            tools.append(
                (
                    ref.name,
                    build_bound_tool(
                        self.registry,
                        configured[ref.name],
                        ref.params,
                        inject,
                    ),
                ),
            )
        return ToolHarvest(tools=tuple(tools))
