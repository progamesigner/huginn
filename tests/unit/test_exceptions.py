import httpx
import pytest
from starlette.types import ASGIApp

from app.core.exceptions import (
    AppError,
    InternalServerError,
    handle_app_error,
    handle_unexpected_error,
)
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
            AppError: handle_app_error,
            Exception: handle_unexpected_error,
        },
    )
    manager.include(router)
    return manager.app


def test_reason_is_derived_from_class_name() -> None:
    class ConversationNotFoundError(AppError):
        status = 404
        code = 9001

    assert ConversationNotFoundError.reason == 'CONVERSATION_NOT_FOUND'


def test_reason_derivation_handles_acronyms() -> None:
    class LLMTimeoutError(AppError):
        status = 504
        code = 9002

    assert LLMTimeoutError.reason == 'LLM_TIMEOUT'


def test_reason_can_be_overridden_explicitly() -> None:
    class WeirdError(AppError):
        status = 400
        code = 9003
        reason = 'CUSTOM_REASON'

    assert WeirdError.reason == 'CUSTOM_REASON'


def test_duplicate_code_raises_at_class_creation() -> None:
    class FirstError(AppError):
        status = 400
        code = 9004

    with pytest.raises(TypeError):

        class SecondError(AppError):
            status = 400
            code = 9004


def test_codespace_accepts_codes_within_range() -> None:
    class FeatureError(AppError):
        codespace = range(9100, 9200)

    class InRangeError(FeatureError):
        status = 400
        code = 9150

    assert InRangeError.code == 9150


def test_codespace_rejects_codes_outside_range() -> None:
    class OtherFeatureError(AppError):
        codespace = range(9200, 9300)

    with pytest.raises(TypeError):

        class OutOfRangeError(OtherFeatureError):
            status = 400
            code = 9999


def test_subclass_without_codespace_is_unrestricted() -> None:
    class UnrestrictedError(AppError):
        status = 400
        code = 9400

    assert UnrestrictedError.code == 9400


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
        'extra': None,
    }


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
        'extra': None,
    }
    assert 'leaked secret' not in response.text
