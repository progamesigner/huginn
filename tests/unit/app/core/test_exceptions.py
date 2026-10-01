from app.core.codespaces import Code
from app.core.exceptions import AppError, InternalServerError, UnauthorizedError


def test_code_carries_both_the_number_and_the_error_name() -> None:
    assert int(Code.CONVERSATION_BUSY) == 3
    assert Code.CONVERSATION_BUSY.name == 'CONVERSATION_BUSY'


def test_existing_codes_keep_their_numbers_and_names() -> None:
    assert InternalServerError.code == 0
    assert InternalServerError.code.name == 'UNKNOWN_ERROR'
    assert UnauthorizedError.code == 1
    assert UnauthorizedError.code.name == 'UNAUTHORIZED'


def test_app_error_keeps_its_message_and_headers() -> None:
    class ExampleError(AppError):
        status = 418
        code = Code.UNKNOWN_ERROR

    error = ExampleError('teapot', headers={'Retry-After': '1'})

    assert str(error) == 'teapot'
    assert error.message == 'teapot'
    assert error.headers == {'Retry-After': '1'}
