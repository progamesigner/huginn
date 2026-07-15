import httpx
from starlette.types import ASGIApp

from app.core.exceptions import AppError, InternalServerError
from app.core.renderers import render_app_error, render_unexpected_error
from lib.router import Router, RouterManager


async def fetch(
    app: ASGIApp,
    path: str,
    *,
    raise_app_exceptions: bool = True,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://huginn',
    ) as client:
        return await client.get(path)


def app_with(router: Router) -> ASGIApp:
    manager = RouterManager(
        exception_handlers={
            AppError: render_app_error,
            Exception: render_unexpected_error,
        },
    )
    manager.include(router)
    return manager.app


async def test_app_error_becomes_envelope_response() -> None:
    class ExampleError(AppError):
        status = 418
        code = 9500

    router = Router()

    @router.get('/boom')
    async def boom() -> None:
        raise ExampleError('teapot', details={'reason': 'because'})

    response = await fetch(app_with(router), '/boom')

    assert response.status_code == 418
    assert response.json() == {
        'code': 9500,
        'reason': 'EXAMPLE',
        'message': 'teapot',
        'details': {'reason': 'because'},
    }


async def test_app_error_headers_are_applied_to_response() -> None:
    class RateLimitedError(AppError):
        status = 429
        code = 9501

    router = Router()

    @router.get('/limited')
    async def limited() -> None:
        raise RateLimitedError('slow down', headers={'Retry-After': '30'})

    response = await fetch(app_with(router), '/limited')

    assert response.status_code == 429
    assert response.headers['retry-after'] == '30'


async def test_app_error_without_headers_omits_extra_headers() -> None:
    class PlainError(AppError):
        status = 400
        code = 9502

    router = Router()

    @router.get('/plain')
    async def plain() -> None:
        raise PlainError('bad request')

    response = await fetch(app_with(router), '/plain')

    assert response.status_code == 400
    assert 'retry-after' not in response.headers


async def test_unexpected_exception_becomes_generic_envelope() -> None:
    router = Router()

    @router.get('/explode')
    async def explode() -> None:
        raise RuntimeError('leaked secret')

    response = await fetch(app_with(router), '/explode', raise_app_exceptions=False)

    assert response.status_code == 500
    body = response.json()
    assert body == {
        'code': InternalServerError.code,
        'reason': 'UNKNOWN_ERROR',
        'message': 'internal server error',
        'details': None,
    }
    assert 'leaked secret' not in response.text
