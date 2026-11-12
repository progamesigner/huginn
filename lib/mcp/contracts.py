from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from httpx2 import URL, AsyncClient
from pydantic import JsonValue

from .types import MCPToolInfo


class MCPClientContract(ABC):
    @abstractmethod
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str: ...

    async def call_tool_result(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        """Prefer structured JSON when available; text-only clients remain valid."""
        return await self.call_tool(server, tool, arguments)

    @abstractmethod
    async def list_tools(self, server: str) -> tuple[MCPToolInfo, ...]: ...

    @abstractmethod
    def unavailable(self) -> tuple[str, ...]: ...


# NOTE: structural so any HTTP client manager fits without lib/mcp importing it
class HTTPConnector(Protocol):
    def connect(
        self,
        base_url: URL | str = '',
        headers: Mapping[str, str] | None = None,
    ) -> AbstractAsyncContextManager[AsyncClient]: ...
