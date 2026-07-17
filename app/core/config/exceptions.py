class ConfigError(Exception):
    pass


class AmbiguousInstructionsError(ConfigError):
    pass


class InvalidValueReferenceError(ConfigError):
    pass


class MissingConfigFileError(ConfigError):
    pass


class MissingEnvironmentVariableError(ConfigError):
    pass


class UnknownAgentReferenceError(ConfigError):
    pass


class UnknownMcpServerReferenceError(ConfigError):
    pass


class UnknownToolReferenceError(ConfigError):
    pass
