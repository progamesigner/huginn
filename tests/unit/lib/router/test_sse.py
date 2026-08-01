import json

from lib.router import ServerSentEvent, format_sse_event


def frame(event: ServerSentEvent) -> str:
    return format_sse_event(
        event=event.event,
        data_str=None if event.data is None else json.dumps(event.data),
        comment=event.comment,
        id=event.id,
    ).decode()


# NOTE: the framing is the vendor's (fastapi ships it since 0.139, so no sse-starlette),
# and these assertions are what pins the wire shape the channel depends on across an
# upgrade -- a named event, JSON data, a comment for keepalive, and a terminator
def test_a_named_event_carries_its_json_payload() -> None:
    assert frame(ServerSentEvent(event='delta', data={'text': 'hi'})) == (
        'event: delta\ndata: {"text": "hi"}\n\n'
    )


def test_a_data_only_event_leaves_the_type_to_the_client() -> None:
    assert frame(ServerSentEvent(data={'text': 'hi'})) == 'data: {"text": "hi"}\n\n'


def test_a_multiline_payload_stays_one_event() -> None:
    # NOTE: a bare newline inside data: would end the event, so JSON escaping it is what
    # keeps a model's multi-line answer from being read as several events
    framed = frame(ServerSentEvent(event='delta', data={'text': 'two\nlines'}))

    assert framed == 'event: delta\ndata: {"text": "two\\nlines"}\n\n'
    assert framed.count('\n\n') == 1


def test_a_comment_frames_a_keepalive_nobody_dispatches_on() -> None:
    assert frame(ServerSentEvent(comment='ping')) == ': ping\n\n'


def test_a_sentinel_payload_may_bypass_json_encoding() -> None:
    # NOTE: the openai-compatible surface ends with `data: [DONE]`, which is not JSON
    assert format_sse_event(data_str='[DONE]').decode() == 'data: [DONE]\n\n'
