from datetime import timedelta

import pytest

from app.exts.telegram.types import MESSAGE, WEBHOOK, Update, WebhookSpec


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


# NOTE: literals on purpose — these tests pin the wire contract, so reading them
# back off the spec would make the assertions vacuous
def test_webhook_spec_pins_the_vendor_wire_contract() -> None:
    assert WEBHOOK.signature_header == 'X-Telegram-Bot-Api-Secret-Token'
    assert WEBHOOK.ack_status == 200
    assert WEBHOOK.reject_status == 403


def test_message_spec_pins_the_vendor_limits() -> None:
    assert MESSAGE.length_limit == 4096
    assert MESSAGE.edit_interval == timedelta(seconds=1.1)


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
