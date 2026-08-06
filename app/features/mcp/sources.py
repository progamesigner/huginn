import logging
from collections.abc import Sequence
from typing import override

from app.features.agents.config import AgentConfig
from app.features.agents.contracts import NamedTool, ToolSource
from lib.mcp import MCPClientContract, MCPError

from .bridge import build_mcp_tools

logger = logging.getLogger(__name__)


class MCPToolSource(ToolSource):
    client: MCPClientContract

    def __init__(self, client: MCPClientContract) -> None:
        self.client = client

    @override
    async def collect(self, agent: AgentConfig) -> Sequence[NamedTool]:
        collected = []
        for ref in agent.mcp:
            try:
                infos = await self.client.list_tools(ref.server)
            except MCPError as error:
                logger.warning(
                    'agent references mcp server %r which is unavailable;'
                    ' skipping (%s)',
                    ref.server,
                    error,
                )
                continue

            tools = build_mcp_tools(
                self.client,
                ref.server,
                infos,
                ref.allow,
                ref.inject,
            )
            if not tools:
                logger.warning(
                    'mcp server %r contributed no tools for this agent (allow=%s)',
                    ref.server,
                    list(ref.allow),
                )
            collected.extend((tool.name, tool) for tool in tools)

        return collected
