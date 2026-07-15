import re
from typing import ClassVar

from pydantic import JsonValue

_REASON_BOUNDARY = re.compile(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')


class BuildError(Exception):
    pass


class AppError(Exception):
    claimed: ClassVar[dict[int, type[AppError]]] = {}

    status: ClassVar[int]
    code: ClassVar[int]
    reason: ClassVar[str]
    codespace: ClassVar[range]

    message: str
    details: JsonValue
    headers: dict[str, str] | None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if 'reason' not in cls.__dict__:
            cls.reason = _REASON_BOUNDARY.sub(
                '_',
                cls.__name__.removesuffix('Error'),
            ).upper()

        if 'code' in cls.__dict__:
            if hasattr(cls, 'codespace') and cls.code not in cls.codespace:
                raise TypeError(
                    f'{cls.__name__} code {cls.code} outside {cls.codespace}'
                )

            if (owner := AppError.claimed.setdefault(cls.code, cls)) is not cls:
                raise TypeError(
                    f'{cls.__name__} reuses code {cls.code} of {owner.__name__}'
                )

    def __init__(
        self,
        message: str,
        details: JsonValue = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details
        self.headers = headers


class InternalServerError(AppError):
    status = 500
    code = 0
    reason = 'UNKNOWN_ERROR'
