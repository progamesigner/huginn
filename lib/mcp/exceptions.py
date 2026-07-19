class MCPError(Exception):
    pass


class MCPConnectionError(MCPError):
    pass


class UnknownMCPServerError(MCPError):
    pass
