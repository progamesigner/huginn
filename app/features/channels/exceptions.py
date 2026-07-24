from app.core.config.exceptions import ConfigError


class DuplicateChannelKindError(ConfigError):
    pass


class UnknownChannelKindError(ConfigError):
    pass


class DuplicateChannelSecretError(ConfigError):
    pass
