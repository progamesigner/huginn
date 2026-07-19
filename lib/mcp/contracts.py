from abc import ABC, abstractmethod
from collections.abc import Mapping

from pydantic import JsonValue

from .types import McpToolInfo


class McpClientContract(ABC):
    @abstractmethod
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str: ...

    @abstractmethod
    async def list_tools(self, server: str) -> tuple[McpToolInfo, ...]: ...
