from typing import ClassVar

from .codespaces import Code


class AppError(Exception):
    status: ClassVar[int]
    code: ClassVar[Code]

    message: str
    headers: dict[str, str] | None

    def __init__(self, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.headers = headers


class InternalServerError(AppError):
    status = 500
    code = Code.UNKNOWN_ERROR


class UnauthorizedError(AppError):
    status = 401
    code = Code.UNAUTHORIZED
