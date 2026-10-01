import json

from .test_native_chat import (
    _CREDENTIAL,
    StreamingTurns,
    api_key,
    chat_client,
    events,
    make_config,
)

_PATH = '/openai/v1/chat/completions'
_AUTH = {'Authorization': f'Bearer {_CREDENTIAL}'}


def payloads(body: str) -> list[str]:
    return [data for _, data in events(body) if data]


async def test_an_openai_stream_carries_chunks_and_a_done_sentinel() -> None:
    turns = StreamingTurns('He', 'llo')

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers=_AUTH,
            json={'model': 'scout', 'messages': [{'role': 'user', 'content': 'hi'}]}
            | {'stream': True},
        )

    assert response.status_code == 200
    assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'
    sent = payloads(response.text)
    assert sent[-1] == '[DONE]'

    first = json.loads(sent[0])
    assert first['object'] == 'chat.completion.chunk'
    assert first['model'] == 'scout'
    assert first['choices'][0]['delta'] == {'role': 'assistant', 'content': 'He'}
    assert json.loads(sent[1])['choices'][0]['delta'] == {'content': 'llo'}
    assert json.loads(sent[2])['choices'][0] == {
        'index': 0,
        'delta': {},
        'finish_reason': 'stop',
    }


# NOTE: this dialect has no way to say 'ran out of time', and 'length' is the nearest
# true thing -- the answer really did stop before the model was finished
async def test_an_openai_stream_that_stopped_short_finishes_on_length() -> None:
    turns = StreamingTurns('He', ending='exhausted')

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers=_AUTH,
            json={'model': 'scout', 'messages': [], 'stream': True},
        )

    sent = payloads(response.text)
    assert sent[-1] == '[DONE]'
    assert json.loads(sent[-2])['choices'][0]['finish_reason'] == 'length'


async def test_every_openai_chunk_shares_one_completion_id() -> None:
    turns = StreamingTurns('He', 'llo')

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers=_AUTH,
            json={'model': 'scout', 'stream': True, 'messages': []},
        )

    identifiers = {
        json.loads(data)['id'] for data in payloads(response.text) if data != '[DONE]'
    }
    assert len(identifiers) == 1
    assert next(iter(identifiers)).startswith('chatcmpl-')


async def test_an_openai_request_without_streaming_answers_in_one_body() -> None:
    turns = StreamingTurns('the ', 'answer')

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers=_AUTH,
            json={'model': 'scout', 'messages': [{'role': 'user', 'content': 'hi'}]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body['object'] == 'chat.completion'
    assert body['choices'][0]['message'] == {
        'role': 'assistant',
        'content': 'the answer',
    }
    assert body['choices'][0]['finish_reason'] == 'stop'
    assert body['id'].startswith('chatcmpl-')
    assert body['created'] > 0

    async def test_an_openai_request_forwards_max_completion_tokens() -> None:
        turns = StreamingTurns()

        async with chat_client(make_config(api_key(agent='scout')), turns) as client:
            response = await client.post(
                _PATH,
                headers=_AUTH,
                json={
                    'model': 'scout',
                    'max_completion_tokens': 4096,
                    'messages': [{'role': 'user', 'content': 'hi'}],
                },
            )

        assert response.status_code == 200
        assert turns.requests[0].max_completion_tokens == 4096


async def test_a_failed_turn_without_streaming_is_a_real_error_status() -> None:
    turns = StreamingTurns('half an ', fail=True)

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers=_AUTH,
            json={'model': 'scout', 'messages': []},
        )

    assert response.status_code == 500
    assert response.json()['error'] == 'TURN_FAILED'


async def test_a_failed_openai_stream_carries_an_error_object() -> None:
    turns = StreamingTurns('half an ', fail=True)

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers=_AUTH,
            json={'model': 'scout', 'stream': True, 'messages': []},
        )

    sent = payloads(response.text)
    assert json.loads(sent[-2])['error']['type'] == 'turn_failed'
    assert sent[-1] == '[DONE]'


async def test_the_model_field_picks_an_agent_the_key_allows() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key()), turns) as client:
        await client.post(
            _PATH,
            headers=_AUTH,
            json={'model': 'concierge', 'stream': True, 'messages': []},
        )

    assert turns.requests[0].policy.agent == 'concierge'


async def test_a_model_that_is_not_an_agent_leaves_the_routing_alone() -> None:
    # NOTE: a drop-in client sends a real model name, which is the agent's business --
    # so it names no agent and the key's own router decides
    turns = StreamingTurns()
    key = api_key(agent='scout')

    async with chat_client(make_config(key), turns) as client:
        response = await client.post(
            _PATH,
            headers=_AUTH,
            json={'model': 'gpt-4o', 'stream': True, 'messages': []},
        )

    assert response.status_code == 200
    assert turns.requests[0].policy.agent == 'scout'


async def test_the_surface_refuses_an_unauthenticated_caller() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(_PATH, json={'model': 'scout', 'messages': []})

    assert response.status_code == 401
    assert turns.requests == []


async def test_a_model_the_key_may_not_use_falls_back_to_its_own_agent() -> None:
    turns = StreamingTurns()
    config = make_config(api_key(agents=('concierge',), agent='concierge'))

    async with chat_client(config, turns) as client:
        response = await client.post(
            _PATH,
            headers=_AUTH,
            json={'model': 'scout', 'messages': [], 'conversation': 'c-1'},
        )

    # NOTE: 'scout' is not an agent this key may use, so it is read as a model name and
    # the key's own pinned agent answers instead
    assert response.status_code == 200
    assert turns.requests[0].policy.agent == 'concierge'
