from app.core.config.exceptions import ConfigError


class ConflictingChannelAgentError(ConfigError):
    pass


class IncompleteChannelRoutingError(ConfigError):
    pass


class DuplicateChannelAgentError(ConfigError):
    pass


class DuplicateChannelKindError(ConfigError):
    pass


class UnknownChannelKindError(ConfigError):
    pass


class DuplicateChannelSecretError(ConfigError):
    pass


class IncompleteTranscriptionError(ConfigError):
    pass


class ChannelError(Exception):
    pass


class DuplicateCommandError(ChannelError):
    pass


class TranscriptionUnavailableError(ChannelError):
    pass


class MissingWebhookURLError(ChannelError):
    pass
