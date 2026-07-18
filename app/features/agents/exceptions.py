from app.core.config.exceptions import ConfigError


class AgentError(Exception):
    pass


class TurnTimeout(AgentError):
    pass


class AmbiguousInstructionsError(ConfigError):
    pass


class UnknownAgentReferenceError(ConfigError):
    pass


class UnknownMcpServerReferenceError(ConfigError):
    pass


class UnknownToolReferenceError(ConfigError):
    pass
