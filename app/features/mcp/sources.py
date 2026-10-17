from typing import override

from app.features.agents.config import AgentConfig
from app.features.agents.contracts import ToolHarvest, ToolSource
from app.features.agents.entities import TurnScope, resolve_bindings
from lib.mcp import MCPClientContract, MCPError
from lib.observability import Logger

from .bridge import build_mcp_tools


class MCPToolSource(ToolSource):
    logger: Logger
    client: MCPClientContract

    def __init__(
        self,
        client: MCPClientContract,
        *,
        logger: Logger,
    ) -> None:
        self.logger = logger.bind(component='mcp')
        self.client = client

    @override
    async def collect(self, agent: AgentConfig, scope: TurnScope) -> ToolHarvest:
        collected = []
        unreachable = []
        for ref in agent.mcp:
            inject = resolve_bindings(ref.inject, ref.bind, scope)
            if inject is None:
                self.logger.info(
                    'mcp server %r omitted because bound keys are unavailable: %s',
                    ref.server,
                    ', '.join(sorted(ref.bind)),
                )
                continue
            try:
                infos = await self.client.list_tools(ref.server)
            except MCPError as error:
                self.logger.warning(
                    'agent references mcp server %r which is unavailable;'
                    ' skipping (%s)',
                    ref.server,
                    error,
                )
                unreachable.append(ref.server)
                continue

            tools = build_mcp_tools(
                self.client,
                ref.server,
                infos,
                ref.allow,
                inject,
            )
            if not tools:
                self.logger.warning(
                    'mcp server %r contributed no tools for this agent (allow=%s)',
                    ref.server,
                    list(ref.allow),
                )
            collected.extend(
                (tool.name, tool) for tool in tools if not tool.writes_brief
            )

        return ToolHarvest(tools=tuple(collected), unreachable=tuple(unreachable))
