import json

from .test_native_chat import (
    _CREDENTIAL,
    StreamingTurns,
    api_key,
    chat_client,
    events,
    make_config,
)

_PATH = '/anthropic/v1/messages'
_AUTH = {'Authorization': f'Bearer {_CREDENTIAL}'}


def payloads(body: str) -> list[str]:
    return [data for _, data in events(body) if data]


async def test_an_anthropic_stream_opens_and_closes_its_message() -> None:
    turns = StreamingTurns('He', 'llo')

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers=_AUTH,
            json={
                'model': 'scout',
                'max_tokens': 64,
                'stream': True,
                'messages': [{'role': 'user', 'content': 'hi'}],
            },
        )

    assert response.status_code == 200
    assert [name for name, _ in events(response.text)] == [
        'message_start',
        'content_block_start',
        'content_block_delta',
        'content_block_delta',
        'content_block_stop',
        'message_delta',
        'message_stop',
    ]


async def test_an_anthropic_delta_carries_the_text_the_sdk_reads() -> None:
    turns = StreamingTurns('He')

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers=_AUTH,
            json={'model': 'scout', 'stream': True, 'messages': []},
        )

    framed = dict(events(response.text))
    assert json.loads(framed['content_block_delta'])['delta'] == {
        'type': 'text_delta',
        'text': 'He',
    }
    assert json.loads(framed['message_start'])['message']['id'].startswith('msg-')
    assert json.loads(framed['message_delta'])['delta']['stop_reason'] == 'end_turn'


async def test_an_anthropic_request_without_streaming_answers_in_one_body() -> None:
    turns = StreamingTurns('the ', 'answer')

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers=_AUTH,
            json={'model': 'scout', 'messages': [{'role': 'user', 'content': 'hi'}]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body['type'] == 'message'
    assert body['content'] == [{'type': 'text', 'text': 'the answer'}]
    assert body['stop_reason'] == 'end_turn'
    assert body['id'].startswith('msg-')


async def test_a_failed_anthropic_stream_carries_an_error_event() -> None:
    turns = StreamingTurns('half an ', fail=True)

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers=_AUTH,
            json={'model': 'scout', 'stream': True, 'messages': []},
        )

    framed = dict(events(response.text))
    assert json.loads(framed['error'])['error']['type'] == 'api_error'


async def test_the_surface_refuses_an_unauthenticated_caller() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(_PATH, json={'model': 'scout', 'messages': []})

    assert response.status_code == 401
    assert turns.requests == []
