class ConfigError(Exception):
    pass


class DuplicateSectionError(ConfigError):
    pass


class InvalidConfigError(ConfigError):
    pass


class InvalidValueReferenceError(ConfigError):
    pass


class MissingConfigFileError(ConfigError):
    pass


class MissingEnvironmentVariableError(ConfigError):
    pass


class MissingSectionError(ConfigError):
    pass


class UnknownCacheControlModelError(ConfigError):
    pass


class UnknownSectionError(ConfigError):
    pass
