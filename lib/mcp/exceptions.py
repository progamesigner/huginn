class McpError(Exception):
    pass


class McpConnectionError(McpError):
    pass


class UnknownMcpServerError(McpError):
    pass
