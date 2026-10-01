from pydantic import JsonValue

from lib.router import JSONResponse, Request, RequestValidationError, Response

from .codespaces import Code
from .exceptions import AppError, InternalServerError

_INVALID_STATUS = 422


def _envelope(code: Code, message: str, request: Request) -> dict[str, JsonValue]:
    return {
        'code': int(code),
        'error': code.name,
        'message': message,
        'request_id': getattr(request.state, 'request_id', None),
    }


async def render_app_error(request: Request, error: AppError) -> Response:
    return JSONResponse(
        status_code=error.status,
        content=_envelope(error.code, error.message, request),
        headers=error.headers,
    )


# NOTE: the only envelope with details; it names what was wrong, never what arrived
async def render_validation_error(
    request: Request,
    error: RequestValidationError,
) -> Response:
    details = [
        {
            'loc': list(issue.get('loc', ())),
            'msg': str(issue.get('msg', '')),
            'type': str(issue.get('type', '')),
        }
        for issue in error.errors()
    ]
    return JSONResponse(
        status_code=_INVALID_STATUS,
        content={
            **_envelope(Code.INVALID_REQUEST, 'request is invalid', request),
            'details': details,
        },
    )


async def render_unexpected_error(request: Request, error: Exception) -> Response:
    return await render_app_error(request, InternalServerError('internal server error'))
