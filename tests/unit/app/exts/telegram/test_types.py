from datetime import timedelta

import pytest
from pydantic import ValidationError

from app.exts.telegram.types import (
    MEDIA,
    MESSAGE,
    TYPING,
    WEBHOOK,
    PhotoSize,
    Update,
    WebhookSpec,
    select_photo,
)


def test_update_ignores_unknown_vendor_fields() -> None:
    update = Update.model_validate(
        {
            'update_id': 42,
            'new_update_field': 'ignored',
            'message': {
                'message_id': 7,
                'date': 1_721_600_000,
                'chat': {'id': 123, 'type': 'private', 'new_chat_field': True},
                'from': {
                    'id': 456,
                    'is_bot': False,
                    'first_name': 'Han',
                    'language_code': 'zh-hant',
                },
                'text': 'hello',
                'new_message_field': {'anything': 'ignored'},
            },
        },
    )

    assert update.update_id == 42
    assert update.message is not None
    assert update.message.from_ is not None
    assert update.message.from_.first_name == 'Han'
    assert update.message.text == 'hello'


def test_the_reader_keeps_the_rejected_input_out_of_its_error_text() -> None:
    # NOTE: pydantic quotes the offending input by default, which puts what somebody typed
    # into every line that prints a validation failure -- the payload reaches a log only
    # when a caller deliberately asks for it
    with pytest.raises(ValidationError) as raised:
        Update.model_validate({'update_id': 42, 'message': {'text': 'my private note'}})

    assert 'my private note' not in str(raised.value)


# NOTE: literals on purpose — these tests pin the wire contract, so reading them
# back off the spec would make the assertions vacuous
def test_webhook_spec_pins_the_vendor_wire_contract() -> None:
    assert WEBHOOK.signature_header == 'X-Telegram-Bot-Api-Secret-Token'
    assert WEBHOOK.ack_status == 200
    assert WEBHOOK.reject_status == 403
    assert WEBHOOK.path == '/webhooks/telegram'


def test_message_spec_pins_the_vendor_limits() -> None:
    assert MESSAGE.length_limit == 4096
    assert MESSAGE.edit_interval == timedelta(seconds=1.1)


def test_media_spec_pins_the_vendor_limits() -> None:
    assert MEDIA.size_limit == 5 * 1024 * 1024


def test_typing_spec_pins_the_vendor_limits() -> None:
    assert TYPING.action == 'typing'
    # NOTE: telegram expires the status after about five seconds, so a refresh at or
    # over that would leave the chat blinking between typing and idle
    assert TYPING.refresh_interval == timedelta(seconds=4)
    assert TYPING.refresh_interval < timedelta(seconds=5)


def photo(width: int, height: int, file_size: int | None) -> PhotoSize:
    return PhotoSize(
        file_id=f'photo-{width}x{height}',
        file_unique_id=f'unique-{width}x{height}',
        width=width,
        height=height,
        file_size=file_size,
    )


def test_select_photo_takes_the_largest_size_that_fits() -> None:
    sizes = (
        photo(90, 60, 900),
        photo(1280, 960, 400_000),
        photo(3000, 2000, 9_000_000),
    )

    selected = select_photo(sizes, limit=1_000_000)

    assert selected is not None
    assert selected.file_id == 'photo-1280x960'


def test_select_photo_ignores_wire_order() -> None:
    sizes = (photo(1280, 960, 400_000), photo(90, 60, 900))

    selected = select_photo(sizes, limit=1_000_000)

    assert selected is not None
    assert selected.file_id == 'photo-1280x960'


def test_select_photo_keeps_an_unmeasured_size_as_a_candidate() -> None:
    selected = select_photo((photo(1280, 960, None),), limit=1_000)

    assert selected is not None
    assert selected.file_id == 'photo-1280x960'


def test_select_photo_without_any_size_that_fits() -> None:
    assert select_photo((photo(3000, 2000, 9_000_000),), limit=1_000_000) is None


def test_select_photo_without_photos() -> None:
    assert select_photo(()) is None


def test_specs_are_frozen_so_a_consumer_cannot_rewrite_a_vendor_fact() -> None:
    with pytest.raises(AttributeError):
        WEBHOOK.ack_status = 204  # ty: ignore[invalid-assignment]


def test_webhook_spec_carries_no_behaviour() -> None:
    methods = [
        name
        for name in vars(WebhookSpec)
        if callable(getattr(WebhookSpec, name)) and not name.startswith('__')
    ]

    assert not methods
