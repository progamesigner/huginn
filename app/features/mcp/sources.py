import logging
from typing import override

from app.features.agents.config import AgentConfig
from app.features.agents.contracts import ToolHarvest, ToolSource
from app.features.agents.entities import TurnScope, resolve_bindings
from lib.mcp import MCPClientContract, MCPError

from .bridge import build_mcp_tools

logger = logging.getLogger(__name__)


class MCPToolSource(ToolSource):
    client: MCPClientContract

    def __init__(self, client: MCPClientContract) -> None:
        self.client = client

    @override
    async def collect(self, agent: AgentConfig, scope: TurnScope) -> ToolHarvest:
        collected = []
        unreachable = []
        for ref in agent.mcp:
            inject = resolve_bindings(ref.inject, ref.bind, scope)
            if inject is None:
                logger.info(
                    'mcp server %r omitted because bound keys are unavailable: %s',
                    ref.server,
                    ', '.join(sorted(ref.bind)),
                )
                continue
            try:
                infos = await self.client.list_tools(ref.server)
            except MCPError as error:
                logger.warning(
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
                logger.warning(
                    'mcp server %r contributed no tools for this agent (allow=%s)',
                    ref.server,
                    list(ref.allow),
                )
            collected.extend((tool.name, tool) for tool in tools)

        return ToolHarvest(tools=tuple(collected), unreachable=tuple(unreachable))
