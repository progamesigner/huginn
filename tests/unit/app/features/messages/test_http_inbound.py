import json
from base64 import b64encode

import pytest

from app.features.messages import handlers
from tests.helpers.blobs import MemoryBlobs

from .test_http_send import _OTHER, bearer, two_keys
from .test_native_chat import StreamingTurns, api_key, chat_client, make_config

_PNG = b'\x89PNG\r\n\x1a\n not really a png'
_CSV = b'name,qty\napple,3\n'


def inline(data: bytes, **extra: object) -> dict[str, object]:
    return {'data': b64encode(data).decode(), **extra}


async def test_an_upload_is_typed_by_its_bytes_not_its_label() -> None:
    blobs = MemoryBlobs()

    async with chat_client(
        make_config(api_key(agent='scout')),
        StreamingTurns(),
        blobs=blobs,
    ) as client:
        response = await client.post(
            '/v1/blobs',
            headers=bearer(),
            files={'file': ('photo.txt', _PNG, 'text/plain')},
        )

    assert response.status_code == 201
    body = response.json()
    assert (body['kind'], body['mime'], body['size'], body['name']) == (
        'image',
        'image/png',
        len(_PNG),
        'photo.txt',
    )
    stored = next(iter(blobs.blobs.values()))
    assert str(stored.id) == body['id']
    assert blobs.ttls[stored.id] is not None
    assert (stored.id, 'web') in blobs.grants


async def test_an_upload_of_a_type_nothing_reads_is_refused() -> None:
    async with chat_client(
        make_config(api_key(agent='scout')),
        StreamingTurns(),
    ) as client:
        response = await client.post(
            '/v1/blobs',
            headers=bearer(),
            files={'file': ('archive.zip', b'PK\x03\x04 zipped', 'application/zip')},
        )

    assert response.status_code == 415
    assert response.json()['error'] == 'UNSUPPORTED_ATTACHMENT'


async def test_an_upload_over_the_limit_is_refused_by_its_declared_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(handlers, '_UPLOAD_LIMIT', 64)
    blobs = MemoryBlobs()

    async with chat_client(
        make_config(api_key(agent='scout')),
        StreamingTurns(),
        blobs=blobs,
    ) as client:
        response = await client.post(
            '/v1/blobs',
            headers=bearer(),
            files={'file': ('big.png', _PNG * 10, 'image/png')},
        )

    assert response.status_code == 413
    assert response.json()['error'] == 'ATTACHMENT_TOO_LARGE'
    assert blobs.blobs == {}


async def test_an_upload_without_a_file_field_is_a_bad_request() -> None:
    async with chat_client(
        make_config(api_key(agent='scout')),
        StreamingTurns(),
    ) as client:
        response = await client.post(
            '/v1/blobs',
            headers=bearer(),
            files={'other': ('x.png', _PNG, 'image/png')},
        )

    assert response.status_code == 400
    assert response.json()['error'] == 'INVALID_ATTACHMENT'


async def test_an_uploaded_blob_rides_along_with_the_native_chat() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        uploaded = await client.post(
            '/v1/blobs',
            headers=bearer(),
            files={'file': ('photo.png', _PNG, 'image/png')},
        )
        response = await client.post(
            '/v1/chat',
            headers=bearer(),
            json={
                'message': 'what is this?',
                'attachments': [{'blob': uploaded.json()['id']}],
            },
        )

    assert response.status_code == 200
    (attachment,) = turns.requests[0].prompts[0].attachments
    assert (attachment.kind, attachment.mime, attachment.data) == (
        'image',
        'image/png',
        _PNG,
    )
    assert attachment.reference == uploaded.json()['id']


async def test_an_inline_image_needs_no_upload() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            '/v1/chat',
            headers=bearer(),
            json={'message': '', 'attachments': [inline(_PNG, mime='image/gif')]},
        )

    assert response.status_code == 200
    (attachment,) = turns.requests[0].prompts[0].attachments
    assert (attachment.kind, attachment.mime) == ('image', 'image/png')
    assert turns.requests[0].prompts[0].text == ''


async def test_an_inline_document_is_read_before_the_turn() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        await client.post(
            '/v1/chat',
            headers=bearer(),
            json={'message': 'sum it', 'attachments': [inline(_CSV, name='stock.csv')]},
        )

    (attachment,) = turns.requests[0].prompts[0].attachments
    assert (attachment.kind, attachment.mime, attachment.name) == (
        'document',
        'text/csv',
        'stock.csv',
    )
    assert attachment.transcript is not None
    assert 'apple' in attachment.transcript
    assert attachment.document_text is not None


async def test_a_remote_file_is_refused_and_pointed_at_the_upload() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            '/v1/chat',
            headers=bearer(),
            json={'message': 'x', 'attachments': [{'url': 'https://169.254.169.254/'}]},
        )

    assert response.status_code == 400
    assert response.json()['error'] == 'INVALID_ATTACHMENT'
    assert '/v1/blobs' in response.json()['message']
    assert turns.requests == []


async def test_another_keys_blob_is_not_found() -> None:
    turns = StreamingTurns()

    async with chat_client(two_keys(), turns) as client:
        uploaded = await client.post(
            '/v1/blobs',
            headers=bearer(_OTHER),
            files={'file': ('photo.png', _PNG, 'image/png')},
        )
        response = await client.post(
            '/v1/chat',
            headers=bearer(),
            json={'message': 'x', 'attachments': [{'blob': uploaded.json()['id']}]},
        )

    assert response.status_code == 404
    assert response.json()['error'] == 'BLOB_NOT_FOUND'
    assert turns.requests == []


@pytest.mark.parametrize(
    ('attachment', 'status', 'error'),
    [
        ({'data': 'not base64!!'}, 400, 'INVALID_ATTACHMENT'),
        ({}, 400, 'INVALID_ATTACHMENT'),
        (inline(b'\x00\x01 binary'), 415, 'UNSUPPORTED_ATTACHMENT'),
        (inline(_PNG + b'0' * (1024 * 1024)), 413, 'ATTACHMENT_TOO_LARGE'),
    ],
)
async def test_an_unusable_inline_attachment_is_refused_before_the_stream(
    attachment: dict[str, object],
    status: int,
    error: str,
) -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            '/v1/chat',
            headers=bearer(),
            json={'message': 'x', 'attachments': [attachment]},
        )

    assert (response.status_code, response.json()['error']) == (status, error)
    assert turns.requests == []


async def test_a_voice_note_nobody_can_hear_leaves_nothing_to_answer() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            '/v1/chat',
            headers=bearer(),
            json={'message': '', 'attachments': [inline(b'OggS voice')]},
        )

    assert response.status_code == 400
    assert turns.requests == []


async def test_an_openai_image_part_reaches_the_turn() -> None:
    turns = StreamingTurns()
    uri = f'data:image/png;base64,{b64encode(_PNG).decode()}'

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            '/openai/v1/chat/completions',
            headers=bearer(),
            json={
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': 'what is this?'},
                            {'type': 'image_url', 'image_url': {'url': uri}},
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    prompt = turns.requests[0].prompts[0]
    assert prompt.text == 'what is this?'
    assert [attachment.kind for attachment in prompt.attachments] == ['image']


async def test_an_openai_image_link_is_refused_rather_than_dropped() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            '/openai/v1/chat/completions',
            headers=bearer(),
            json={
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {'url': 'https://example.com/a.png'},
                            },
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 400
    assert turns.requests == []


async def test_an_openai_file_part_arrives_as_a_document() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        await client.post(
            '/openai/v1/chat/completions',
            headers=bearer(),
            json={
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'file',
                                'file': {
                                    'filename': 'stock.csv',
                                    'file_data': b64encode(_CSV).decode(),
                                },
                            },
                        ],
                    },
                ],
            },
        )

    (attachment,) = turns.requests[0].prompts[0].attachments
    assert (attachment.kind, attachment.name) == ('document', 'stock.csv')


async def test_an_anthropic_image_block_reaches_the_turn() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            '/anthropic/v1/messages',
            headers=bearer(),
            json={
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image',
                                'source': {
                                    'type': 'base64',
                                    'media_type': 'image/png',
                                    'data': b64encode(_PNG).decode(),
                                },
                            },
                            {'type': 'text', 'text': 'and this?'},
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    prompt = turns.requests[0].prompts[0]
    assert prompt.text == 'and this?'
    assert [attachment.mime for attachment in prompt.attachments] == ['image/png']


async def test_an_anthropic_url_source_is_refused() -> None:
    async with chat_client(
        make_config(api_key(agent='scout')),
        StreamingTurns(),
    ) as client:
        response = await client.post(
            '/anthropic/v1/messages',
            headers=bearer(),
            json={
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'document',
                                'source': {
                                    'type': 'url',
                                    'url': 'https://x.example/a.pdf',
                                },
                            },
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 400


async def test_a_plain_text_request_carries_no_attachments() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        await client.post('/v1/chat', headers=bearer(), json={'message': 'hi'})

    assert turns.requests[0].prompts[0].attachments == ()
    assert json.loads(turns.requests[0].model_dump_json())['prompts'][0]['text'] == 'hi'
