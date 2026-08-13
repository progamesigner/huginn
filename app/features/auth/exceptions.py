from app.core.config.exceptions import ConfigError


class DuplicateApiKeyTokenError(ConfigError):
    pass


class EmptyApiKeyTokenError(ConfigError):
    pass


class ConflictingApiKeyAgentError(ConfigError):
    pass


class IncompleteApiKeyRoutingError(ConfigError):
    pass


class DuplicateApiKeyAgentError(ConfigError):
    pass


class MissingApiKeyAgentsError(ConfigError):
    pass
