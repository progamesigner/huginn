from app.features.messages.entities import (
    DoneEvent,
    ErrorEvent,
    OpenedEvent,
    PingEvent,
    RoutedEvent,
    TextEvent,
    ToolEvent,
    TurnEvent,
    UnfinishedEvent,
)
from app.features.messages.presenters import NativePresenter, encode


def framed(event: TurnEvent) -> str:
    return ''.join(encode(presented) for presented in NativePresenter().present(event))


def test_the_opening_event_names_the_conversation_to_continue() -> None:
    # NOTE: a client that sent no conversation id learns here which one it was given,
    # and that has to arrive before anything it might want to store the answer under
    assert framed(OpenedEvent(conversation='c-1')) == (
        'event: opened\ndata: {"conversation": "c-1"}\n\n'
    )


def test_the_routed_event_names_the_agent() -> None:
    assert framed(RoutedEvent(agent='scout')) == (
        'event: routed\ndata: {"agent": "scout"}\n\n'
    )


def test_a_text_fragment_becomes_a_delta() -> None:
    assert framed(TextEvent(text='He')) == 'event: delta\ndata: {"text": "He"}\n\n'


def test_a_tool_step_carries_its_name_and_phase() -> None:
    assert framed(ToolEvent(name='websearch', phase='started')) == (
        'event: tool\ndata: {"name": "websearch", "phase": "started"}\n\n'
    )


def test_the_done_event_closes_the_turn() -> None:
    assert framed(DoneEvent()) == 'event: done\ndata: {}\n\n'


def test_an_unfinished_event_names_why_the_answer_stops_short() -> None:
    # NOTE: distinct from `error` on purpose -- the answer is real and worth showing, it
    # just stops earlier than the reader expects
    assert framed(UnfinishedEvent(ending='timeout')) == (
        'event: unfinished\ndata: {"ending": "timeout"}\n\n'
    )


def test_an_error_event_carries_its_message() -> None:
    assert framed(ErrorEvent(message='it broke')) == (
        'event: error\ndata: {"message": "it broke"}\n\n'
    )


def test_a_ping_is_a_comment_no_client_dispatches_on() -> None:
    assert framed(PingEvent()) == ': ping\n\n'


def test_a_fragment_keeps_its_own_alphabet() -> None:
    # NOTE: json escaping every CJK character would treble the size of a Chinese answer
    assert framed(TextEvent(text='你好')) == 'event: delta\ndata: {"text": "你好"}\n\n'


def test_a_multiline_fragment_stays_one_frame() -> None:
    framed_text = framed(TextEvent(text='one\ntwo'))

    assert framed_text == 'event: delta\ndata: {"text": "one\\ntwo"}\n\n'
    assert framed_text.count('\n\n') == 1
