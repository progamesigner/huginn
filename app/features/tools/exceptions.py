from app.core.config.exceptions import ConfigError


class ToolError(Exception):
    pass


class DuplicateToolError(ToolError):
    pass


class UnknownToolError(ToolError):
    pass


class InvalidToolOverlayError(ConfigError):
    pass


class UnknownToolKindError(ConfigError):
    pass


class UnknownToolReferenceError(ConfigError):
    pass
