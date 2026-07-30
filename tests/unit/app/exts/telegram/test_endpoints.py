import json

from app.exts.telegram.endpoints import _inbound


def body(**message: object) -> bytes:
    return json.dumps(
        {
            'update_id': 42,
            'message': {
                'message_id': 7,
                'date': 1_721_600_000,
                'chat': {'id': 123, 'type': 'private'},
                'from': {'id': 456, 'is_bot': False, 'first_name': 'Han'},
                **message,
            },
        },
    ).encode()


def photo(width: int, height: int, file_size: int) -> dict[str, object]:
    return {
        'file_id': f'photo-{width}x{height}',
        'file_unique_id': f'unique-{width}x{height}',
        'width': width,
        'height': height,
        'file_size': file_size,
    }


def test_inbound_carries_no_attachment_for_a_text_message() -> None:
    message = _inbound('support', body(text='hello'))

    assert message is not None
    assert message.text == 'hello'
    assert message.attachments == ()


def test_inbound_carries_the_photo_size_that_fits() -> None:
    message = _inbound(
        'support',
        body(
            caption='what is this',
            photo=[photo(90, 60, 900), photo(1280, 960, 400_000)],
        ),
    )

    assert message is not None
    assert message.text == 'what is this'
    assert len(message.attachments) == 1
    attachment = message.attachments[0]
    assert attachment.kind == 'image'
    assert attachment.mime == 'image/jpeg'
    assert attachment.reference == 'photo-1280x960'
    assert attachment.unique == 'unique-1280x960'
    assert (attachment.width, attachment.height) == (1280, 960)
    assert attachment.size == 400_000


def test_inbound_carries_a_photo_without_a_caption() -> None:
    message = _inbound('support', body(photo=[photo(1280, 960, 400_000)]))

    assert message is not None
    assert message.text == ''
    assert len(message.attachments) == 1


def test_inbound_drops_a_photo_no_size_of_which_fits() -> None:
    message = _inbound('support', body(photo=[photo(6000, 4000, 9_000_000)]))

    assert message is not None
    assert message.attachments == ()


def test_inbound_carries_a_voice_note() -> None:
    message = _inbound(
        'support',
        body(
            voice={
                'file_id': 'voice-file',
                'file_unique_id': 'voice-unique',
                'duration': 12,
                'mime_type': 'audio/ogg',
                'file_size': 8_000,
            },
        ),
    )

    assert message is not None
    assert len(message.attachments) == 1
    attachment = message.attachments[0]
    assert attachment.kind == 'audio'
    assert attachment.mime == 'audio/ogg'
    assert attachment.reference == 'voice-file'
    assert attachment.unique == 'voice-unique'
    assert attachment.duration == 12
    assert attachment.size == 8_000


def test_inbound_falls_back_to_the_vendor_voice_mime() -> None:
    message = _inbound(
        'support',
        body(
            voice={
                'file_id': 'voice-file',
                'file_unique_id': 'voice-unique',
                'duration': 12,
            },
        ),
    )

    assert message is not None
    assert message.attachments[0].mime == 'audio/ogg'


def test_inbound_carries_both_a_photo_and_a_voice_note() -> None:
    message = _inbound(
        'support',
        body(
            photo=[photo(1280, 960, 400_000)],
            voice={
                'file_id': 'voice-file',
                'file_unique_id': 'voice-unique',
                'duration': 3,
            },
        ),
    )

    assert message is not None
    assert [attachment.kind for attachment in message.attachments] == ['image', 'audio']


def test_inbound_discards_an_update_without_a_message() -> None:
    assert _inbound('support', json.dumps({'update_id': 42}).encode()) is None


def test_inbound_discards_an_unparseable_update() -> None:
    assert _inbound('support', b'not json') is None
