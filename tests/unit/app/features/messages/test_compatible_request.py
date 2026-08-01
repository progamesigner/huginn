import json

from app.features.messages.types import CompatibleRequest


def test_the_latest_user_message_is_the_prompt() -> None:
    # NOTE: a compatible client resends its whole history every turn, but the transcript
    # lives here -- so only what the user just said is new
    ask = CompatibleRequest.model_validate_json(
        json.dumps(
            {
                'model': 'scout',
                'messages': [
                    {'role': 'user', 'content': 'first'},
                    {'role': 'assistant', 'content': 'an answer'},
                    {'role': 'user', 'content': 'second'},
                ],
            },
        ),
    )

    assert ask.latest() == 'second'


def test_a_message_may_arrive_as_content_parts() -> None:
    ask = CompatibleRequest.model_validate_json(
        json.dumps(
            {
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': 'look at '},
                            {'type': 'image', 'text': 'ignored'},
                            {'type': 'text', 'text': 'this'},
                        ],
                    },
                ],
            },
        ),
    )

    assert ask.latest() == 'look at this'


def test_a_request_with_no_user_message_asks_nothing() -> None:
    ask = CompatibleRequest.model_validate_json(
        json.dumps({'messages': [{'role': 'system', 'content': 'be nice'}]}),
    )

    assert ask.latest() == ''


def test_the_thread_prefers_an_explicit_conversation_over_the_user() -> None:
    ask = CompatibleRequest.model_validate_json(
        json.dumps({'conversation': 'c-1', 'user': 'u-1'}),
    )

    assert ask.thread() == 'c-1'


def test_the_user_field_keeps_a_compatible_client_on_one_thread() -> None:
    # NOTE: the openai wire has no conversation id, and `user` is the only field a client
    # already sends that identifies who is talking
    ask = CompatibleRequest.model_validate_json(json.dumps({'user': 'u-1'}))

    assert ask.thread() == 'u-1'
