from app.core.config.exceptions import ConfigError


class DuplicateApiKeyTokenError(ConfigError):
    pass


class EmptyApiKeyTokenError(ConfigError):
    pass


class MissingApiKeyAgentsError(ConfigError):
    pass
