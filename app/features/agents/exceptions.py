from app.core.config.exceptions import ConfigError


class AgentError(Exception):
    pass


class TurnTimeoutError(AgentError):
    pass


class UnroutableError(AgentError):
    pass


class AmbiguousInstructionsError(ConfigError):
    pass


class UnknownAgentReferenceError(ConfigError):
    pass


class UnknownMCPServerReferenceError(ConfigError):
    pass


class UnknownToolReferenceError(ConfigError):
    pass


class InvalidToolOverlayError(ConfigError):
    pass


class MissingModelError(ConfigError):
    pass
