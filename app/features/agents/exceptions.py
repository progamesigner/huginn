from app.core.config.exceptions import ConfigError


class AgentError(Exception):
    pass


class DuplicateToolNameError(AgentError):
    pass


class MalformedToolCallError(AgentError):
    pass


class UnroutableError(AgentError):
    pass


class UnservableModelError(AgentError):
    pass


class AmbiguousInstructionsError(ConfigError):
    pass


class UnknownAgentReferenceError(ConfigError):
    pass


class MissingModelError(ConfigError):
    pass


class UnknownProviderReferenceError(ConfigError):
    pass


class ProviderModelMismatchError(ConfigError):
    pass


class AmbiguousModelError(ConfigError):
    pass
