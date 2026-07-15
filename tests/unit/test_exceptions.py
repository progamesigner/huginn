import pytest

from app.core.exceptions import AppError


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
