from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from lib.llm import (
    FinishChunk,
    Message,
    TextChunk,
    TextSegment,
    ThinkingSegment,
    ToolCallChunk,
    ToolCallSegment,
    ToolResultSegment,
)
from lib.llm.adapters.openai import build_messages, decode_chunk


def make_chunk(*choices: Choice) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id='chunk_1',
        choices=list(choices),
        created=0,
        model='gpt-5',
        object='chat.completion.chunk',
    )


def test_build_messages_flattens_text_segments_into_content() -> None:
    messages = (Message(role='user', segments=(TextSegment(text='hi there'),)),)

    assert list(build_messages(messages)) == [{'role': 'user', 'content': 'hi there'}]


def test_build_messages_collects_tool_call_segments() -> None:
    messages = (
        Message(
            role='assistant',
            segments=(ToolCallSegment(id='call_1', name='search', arguments='{}'),),
        ),
    )

    assert list(build_messages(messages)) == [
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'search', 'arguments': '{}'},
                }
            ],
        }
    ]


def test_build_messages_maps_tool_result_to_tool_role_entry() -> None:
    messages = (
        Message(
            role='tool',
            segments=(ToolResultSegment(tool_call_id='call_1', content='42'),),
        ),
    )

    assert list(build_messages(messages)) == [
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': '42'}
    ]


def test_build_messages_drops_thinking_segments() -> None:
    messages = (
        Message(
            role='assistant',
            segments=(TextSegment(text='ok'), ThinkingSegment(data={'thinking': 'x'})),
        ),
    )

    assert list(build_messages(messages)) == [{'role': 'assistant', 'content': 'ok'}]


def test_build_messages_preserves_order_across_messages() -> None:
    messages = (
        Message(role='user', segments=(TextSegment(text='question'),)),
        Message(
            role='assistant',
            segments=(ToolCallSegment(id='call_1', name='search', arguments='{}'),),
        ),
        Message(
            role='tool',
            segments=(ToolResultSegment(tool_call_id='call_1', content='ok'),),
        ),
    )

    built = build_messages(messages)

    assert [entry['role'] for entry in built] == ['user', 'assistant', 'tool']


def test_decode_chunk_yields_text_delta() -> None:
    chunk = make_chunk(
        Choice(index=0, delta=ChoiceDelta(content='hi'), finish_reason=None)
    )

    assert list(decode_chunk(chunk)) == [TextChunk(text='hi')]


def test_decode_chunk_skips_empty_choices() -> None:
    chunk = make_chunk()

    assert list(decode_chunk(chunk)) == []


def test_decode_chunk_yields_single_tool_call_fragment() -> None:
    delta = ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=0,
                id='call_1',
                function=ChoiceDeltaToolCallFunction(name='search', arguments='{"q'),
                type='function',
            )
        ]
    )
    chunk = make_chunk(Choice(index=0, delta=delta, finish_reason=None))

    assert list(decode_chunk(chunk)) == [
        ToolCallChunk(index=0, id='call_1', name='search', arguments='{"q')
    ]


def test_decode_chunk_yields_interleaved_multi_tool_calls_in_order() -> None:
    delta = ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=0,
                function=ChoiceDeltaToolCallFunction(arguments='"}'),
            ),
            ChoiceDeltaToolCall(
                index=1,
                id='call_2',
                function=ChoiceDeltaToolCallFunction(name='other', arguments=''),
                type='function',
            ),
        ]
    )
    chunk = make_chunk(Choice(index=0, delta=delta, finish_reason=None))

    assert list(decode_chunk(chunk)) == [
        ToolCallChunk(index=0, id=None, name=None, arguments='"}'),
        ToolCallChunk(index=1, id='call_2', name='other', arguments=''),
    ]


def test_decode_chunk_maps_stop_finish_reason() -> None:
    chunk = make_chunk(Choice(index=0, delta=ChoiceDelta(), finish_reason='stop'))

    assert list(decode_chunk(chunk)) == [FinishChunk(reason='stop')]


def test_decode_chunk_maps_length_finish_reason() -> None:
    chunk = make_chunk(Choice(index=0, delta=ChoiceDelta(), finish_reason='length'))

    assert list(decode_chunk(chunk)) == [FinishChunk(reason='length')]


def test_decode_chunk_maps_tool_calls_finish_reason() -> None:
    chunk = make_chunk(Choice(index=0, delta=ChoiceDelta(), finish_reason='tool_calls'))

    assert list(decode_chunk(chunk)) == [FinishChunk(reason='tool_calls')]


def test_decode_chunk_maps_deprecated_function_call_finish_reason_to_tool_calls() -> (
    None
):
    chunk = make_chunk(
        Choice(index=0, delta=ChoiceDelta(), finish_reason='function_call')
    )

    assert list(decode_chunk(chunk)) == [FinishChunk(reason='tool_calls')]


def test_decode_chunk_maps_content_filter_finish_reason_to_error() -> None:
    chunk = make_chunk(
        Choice(index=0, delta=ChoiceDelta(), finish_reason='content_filter')
    )

    assert list(decode_chunk(chunk)) == [FinishChunk(reason='error')]
