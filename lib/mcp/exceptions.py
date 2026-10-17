class MCPError(Exception):
    pass


class MCPToolError(MCPError):
    def __init__(self, code: str | None = None) -> None:
        super().__init__('MCP tool returned an error')
        self.code = code


class MCPConnectionError(MCPError):
    pass


class UnknownMCPServerError(MCPError):
    pass
