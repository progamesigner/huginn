from app.exts.telegram.types import Update


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
        }
    )

    assert update.update_id == 42
    assert update.message is not None
    assert update.message.from_ is not None
    assert update.message.from_.first_name == 'Han'
    assert update.message.text == 'hello'
