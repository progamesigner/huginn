from typing import override

from app.core.config.registry import HuginnConfig
from app.features.agents.config import AgentConfig
from app.features.agents.contracts import ToolHarvest, ToolSource
from app.features.agents.entities import TurnScope, resolve_bindings
from lib.observability import Logger

from .binding import build_bound_tool
from .config import TOOLS
from .registry import ToolRegistry


class ConfiguredToolSource(ToolSource):
    logger: Logger
    config: HuginnConfig
    registry: ToolRegistry

    def __init__(
        self,
        config: HuginnConfig,
        registry: ToolRegistry,
        *,
        logger: Logger,
    ) -> None:
        self.logger = logger.bind(component='tools')
        self.config = config
        self.registry = registry

    @override
    async def collect(self, agent: AgentConfig, scope: TurnScope) -> ToolHarvest:
        configured = self.config.section(TOOLS).root
        tools = []
        for ref in agent.tools:
            inject = resolve_bindings(ref.inject, ref.bind, scope)
            if inject is None:
                self.logger.info(
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
                        scope,
                    ),
                ),
            )
        return ToolHarvest(tools=tuple(tools))
