from app.core.config.exceptions import ConfigError


class ToolError(Exception):
    pass


class DuplicateToolError(ToolError):
    pass


class UnknownToolError(ToolError):
    pass


class UnknownToolKindError(ConfigError):
    pass
