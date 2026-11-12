import httpx2
from pydantic import BaseModel
from starlette.middleware import Middleware
from starlette.types import ASGIApp

from app.core.codespaces import Code
from app.core.exceptions import AppError
from app.core.middleware import RequestContextMiddleware
from app.core.renderers import (
    render_app_error,
    render_unexpected_error,
    render_validation_error,
)
from lib.router import RequestValidationError, Router, RouterManager


async def fetch(
    app: ASGIApp,
    path: str,
    *,
    raise_app_exceptions: bool = True,
    json: object = None,
) -> httpx2.Response:
    transport = httpx2.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    async with httpx2.AsyncClient(
        transport=transport,
        base_url='http://huginn',
    ) as client:
        if json is not None:
            return await client.post(path, json=json)
        return await client.get(path)


def app_with(router: Router) -> ASGIApp:
    manager = RouterManager(
        exception_handlers={
            AppError: render_app_error,
            RequestValidationError: render_validation_error,
            Exception: render_unexpected_error,
        },
        middlewares=[Middleware(RequestContextMiddleware)],
    )
    manager.include(router)
    return manager.app


async def test_app_error_becomes_envelope_response() -> None:
    class ExampleError(AppError):
        status = 418
        code = Code.TURN_FAILED

    router = Router()

    @router.get('/boom')
    async def boom() -> None:
        raise ExampleError('teapot')

    response = await fetch(app_with(router), '/boom')

    assert response.status_code == 418
    body = response.json()
    request_id = body.pop('request_id')
    assert body == {'code': 5, 'error': 'TURN_FAILED', 'message': 'teapot'}
    assert isinstance(request_id, str)
    assert request_id


async def test_app_error_headers_are_applied_to_response() -> None:
    class RateLimitedError(AppError):
        status = 429
        code = Code.UNKNOWN_ERROR

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
        code = Code.UNKNOWN_ERROR

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
    body.pop('request_id')
    assert body == {
        'code': 0,
        'error': 'UNKNOWN_ERROR',
        'message': 'internal server error',
    }
    assert 'leaked secret' not in response.text


async def test_validation_error_alone_carries_details_without_the_input() -> None:
    class Payload(BaseModel):
        count: int

    router = Router()

    @router.post('/count')
    async def count(payload: Payload) -> int:
        return payload.count

    response = await fetch(app_with(router), '/count', json={'count': 'secret-ish'})

    assert response.status_code == 422
    body = response.json()
    assert body['code'] == 6
    assert body['error'] == 'INVALID_REQUEST'
    assert body['details'][0]['loc'] == ['body', 'count']
    assert 'secret-ish' not in response.text
