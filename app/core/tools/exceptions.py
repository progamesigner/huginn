class ToolError(Exception):
    pass


class DuplicateToolError(ToolError):
    pass


class UnknownToolError(ToolError):
    pass
