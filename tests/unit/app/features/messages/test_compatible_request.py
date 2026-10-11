import json

import pytest
from pydantic import ValidationError

from app.features.messages.schemas import CompatibleRequest


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


@pytest.mark.parametrize('field', ['max_completion_tokens', 'max_tokens'])
def test_compatible_token_limit_names_are_normalized(field: str) -> None:
    ask = CompatibleRequest.model_validate({field: 4096})

    assert ask.max_completion_tokens == 4096


def test_canonical_token_limit_wins_when_both_names_are_sent() -> None:
    ask = CompatibleRequest.model_validate(
        {'max_completion_tokens': 8192, 'max_tokens': 4096},
    )

    assert ask.max_completion_tokens == 8192


@pytest.mark.parametrize('value', [0, -1])
def test_compatible_token_limit_must_be_positive(value: int) -> None:
    with pytest.raises(ValidationError):
        CompatibleRequest.model_validate({'max_completion_tokens': value})


def _parts(count: int) -> list[dict[str, object]]:
    return [{'type': 'input_audio', 'input_audio': {'data': 'T2dnUw=='}}] * count


def test_the_latest_message_carries_at_most_ten_attachments() -> None:
    with pytest.raises(ValidationError):
        CompatibleRequest.model_validate(
            {'messages': [{'role': 'user', 'content': _parts(11)}]},
        )


def test_attachments_up_to_the_limit_are_accepted() -> None:
    ask = CompatibleRequest.model_validate(
        {'messages': [{'role': 'user', 'content': _parts(10)}]},
    )

    assert len(ask.latest_attachments()) == 10


def test_attachments_on_earlier_messages_do_not_count() -> None:
    # NOTE: a compatible client resends its history, but only the latest turn is read
    ask = CompatibleRequest.model_validate(
        {
            'messages': [
                {'role': 'user', 'content': _parts(11)},
                {'role': 'user', 'content': 'and now?'},
            ],
        },
    )

    assert ask.latest() == 'and now?'
