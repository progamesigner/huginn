from datetime import timedelta

import pytest
from pydantic import ValidationError

from app.exts.telegram.types import (
    MEDIA,
    MESSAGE,
    TYPING,
    WEBHOOK,
    MessageEntity,
    PhotoSize,
    Update,
    WebhookSpec,
    clip_entities,
    render,
    select_photo,
    utf16_offsets,
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


def test_media_spec_pins_the_vendor_mimes() -> None:
    assert MEDIA.photo_mime == 'image/jpeg'
    assert MEDIA.audio_mime == 'audio/ogg'


def test_typing_spec_pins_the_vendor_limits() -> None:
    assert TYPING.action == 'typing'
    # NOTE: telegram expires the status after about five seconds, so a refresh at or
    # over that would leave the chat blinking between typing and idle
    assert TYPING.refresh_interval == timedelta(seconds=4)
    assert TYPING.refresh_interval < timedelta(seconds=5)


def test_render_maps_markdown_to_plain_text_and_entities() -> None:
    text, entities = render('**bold**')

    assert text == 'bold'
    assert entities == (MessageEntity(type='bold', offset=0, length=4),)


def test_render_measures_offsets_in_utf16_code_units() -> None:
    # NOTE: the globe is outside the bmp and counts as two, each cjk character as one, so
    # a converter measuring python characters would put the bold one unit early
    text, entities = render('🌍 **粗體**')

    assert text == '🌍 粗體'
    assert entities == (MessageEntity(type='bold', offset=3, length=2),)


def test_render_carries_the_fields_each_entity_kind_needs() -> None:
    _, linked = render('[x](https://a)')
    _, fenced = render('```python\nx = 1\n```')

    assert linked == (
        MessageEntity(type='text_link', offset=0, length=1, url='https://a'),
    )
    assert fenced == (MessageEntity(type='pre', offset=0, length=5, language='python'),)


def test_render_leaves_the_bot_api_fields_alone() -> None:
    # NOTE: the converter's own entity also carries `user`/`unix_time`/`date_time_format`,
    # which the bot api does not accept -- reading it field by field is what keeps them out
    _, entities = render('**bold**')

    assert entities[0].model_dump(exclude_none=True) == {
        'type': 'bold',
        'offset': 0,
        'length': 4,
    }


def test_render_returns_no_entity_for_plain_prose() -> None:
    assert render('just prose') == ('just prose', ())


def test_utf16_offsets_counts_up_to_every_character_boundary() -> None:
    assert utf16_offsets('a🌍界') == (0, 1, 3, 4)


def test_utf16_offsets_of_the_empty_string_is_a_single_zero() -> None:
    assert utf16_offsets('') == (0,)


def test_clip_entities_trims_what_straddles_the_window() -> None:
    spanning = (MessageEntity(type='bold', offset=2, length=6),)

    assert clip_entities(spanning, 4, 10) == (
        MessageEntity(type='bold', offset=0, length=4),
    )


def test_clip_entities_rebases_what_the_window_holds_whole() -> None:
    inside = (MessageEntity(type='code', offset=6, length=3),)

    assert clip_entities(inside, 4, 10) == (
        MessageEntity(type='code', offset=2, length=3),
    )


def test_clip_entities_drops_what_falls_outside_the_window() -> None:
    outside = (
        MessageEntity(type='bold', offset=0, length=4),
        MessageEntity(type='italic', offset=10, length=2),
    )

    assert clip_entities(outside, 4, 10) == ()


def test_clip_entities_keeps_the_fields_a_trimmed_entity_still_needs() -> None:
    spanning = (MessageEntity(type='text_link', offset=2, length=6, url='https://a'),)

    assert clip_entities(spanning, 4, 10)[0].url == 'https://a'


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
