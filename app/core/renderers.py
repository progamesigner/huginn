from lib.router import JSONResponse, Request, Response

from .exceptions import AppError, InternalServerError


async def render_app_error(_: Request, error: AppError) -> Response:
    return JSONResponse(
        status_code=error.status,
        content={
            'code': error.code,
            'reason': error.reason,
            'message': error.message,
            'details': error.details,
        },
        headers=error.headers,
    )


async def render_unexpected_error(request: Request, error: Exception) -> Response:
    return await render_app_error(request, InternalServerError('internal server error'))
