import json

import pytest

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


def test_inbound_carries_an_audio_file() -> None:
    message = _inbound(
        'support',
        body(
            audio={
                'file_id': 'audio-file',
                'file_unique_id': 'audio-unique',
                'duration': 154,
                'mime_type': 'audio/mpeg',
                'file_size': 2_000_000,
            },
        ),
    )

    assert message is not None
    assert len(message.attachments) == 1
    attachment = message.attachments[0]
    assert attachment.kind == 'audio'
    assert attachment.mime == 'audio/mpeg'
    assert attachment.reference == 'audio-file'


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


def test_inbound_lands_the_body_of_an_unparseable_update(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # NOTE: "land the raw payload first" is the point of the webhook rule, and a reader
    # that cannot parse an update is the only moment anybody needs the bytes back
    unparseable = b'{"update_id": 42, "message": "a sentence, not an object"}'

    with caplog.at_level('ERROR'):
        assert _inbound('support', unparseable) is None

    assert 'a sentence, not an object' in caplog.text


def entity(offset: int, length: int, type_: str = 'bot_command') -> dict[str, object]:
    return {'type': type_, 'offset': offset, 'length': length}


def test_inbound_reads_a_command_off_the_vendor_entities() -> None:
    message = _inbound('support', body(text='/new', entities=[entity(0, 4)]))

    assert message is not None
    assert message.command is not None
    assert message.command.name == 'new'
    assert message.command.args == ''


def test_inbound_keeps_what_follows_a_command_as_its_arguments() -> None:
    message = _inbound(
        'support',
        body(text='/register  ABC123 ', entities=[entity(0, 9)]),
    )

    assert message is not None
    assert message.command is not None
    assert message.command.name == 'register'
    assert message.command.args == 'ABC123'


def test_inbound_drops_the_bot_mention_from_a_group_command() -> None:
    # NOTE: telegram writes /cmd@botname in groups, and the mention is addressing, not
    # part of the command the app registered
    message = _inbound(
        'support',
        body(text='/help@huginn_bot please', entities=[entity(0, 16)]),
    )

    assert message is not None
    assert message.command is not None
    assert message.command.name == 'help'
    assert message.command.args == 'please'


def test_inbound_ignores_a_command_that_does_not_open_the_message() -> None:
    # NOTE: only the first token invokes; a slash mid-sentence is the user talking
    message = _inbound(
        'support',
        body(text='ask it with /help next time', entities=[entity(12, 5)]),
    )

    assert message is not None
    assert message.command is None
    assert message.text == 'ask it with /help next time'


def test_inbound_ignores_an_entity_that_is_not_a_command() -> None:
    message = _inbound(
        'support',
        body(text='@someone hello', entities=[entity(0, 8, 'mention')]),
    )

    assert message is not None
    assert message.command is None


def test_inbound_lowercases_the_command_name() -> None:
    message = _inbound('support', body(text='/NEW', entities=[entity(0, 4)]))

    assert message is not None
    assert message.command is not None
    assert message.command.name == 'new'


def test_inbound_carries_no_command_for_ordinary_text() -> None:
    message = _inbound('support', body(text='hello'))

    assert message is not None
    assert message.command is None


def test_inbound_names_the_vendor_that_delivered_it() -> None:
    # NOTE: identity is keyed by provider and realm, so a command handler that never
    # sees the channel config still has to learn which vendor this came from
    message = _inbound('support', body(text='hello'))

    assert message is not None
    assert message.kind == 'telegram'
    assert message.channel == 'support'
