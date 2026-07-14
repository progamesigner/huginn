import re
from typing import ClassVar

from pydantic import JsonValue

from lib.router import JSONResponse, Request, Response

_REASON_BOUNDARY = re.compile(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')


class BuildError(Exception):
    pass


class AppError(Exception):
    registry: ClassVar[dict[int, type[AppError]]] = {}

    status: ClassVar[int]
    code: ClassVar[int]
    reason: ClassVar[str]
    codespace: ClassVar[range]

    message: str
    details: JsonValue
    extra: JsonValue

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

            if (registered := AppError.registry.setdefault(cls.code, cls)) is not cls:
                raise TypeError(
                    f'{cls.__name__} reuses code {cls.code} of {registered.__name__}'
                )

    def __init__(
        self,
        message: str,
        details: JsonValue = None,
        extra: JsonValue = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details
        self.extra = extra


class InternalServerError(AppError):
    status = 500
    code = 0
    reason = 'UNKNOWN_ERROR'


async def handle_app_error(_: Request, error: AppError) -> Response:
    return JSONResponse(
        status_code=error.status,
        content={
            'code': error.code,
            'reason': error.reason,
            'message': error.message,
            'details': error.details,
            'extra': error.extra,
        },
    )


async def handle_unexpected_error(request: Request, error: Exception) -> Response:
    return await handle_app_error(request, InternalServerError('internal server error'))
