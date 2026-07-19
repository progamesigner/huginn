from anthropic.types import (
    InputJSONDelta,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
    ThinkingDelta,
    ToolUseBlock,
)
from anthropic.types.raw_message_delta_event import Delta

from lib.llm import (
    FinishChunk,
    Message,
    TextChunk,
    TextSegment,
    ThinkingChunk,
    ThinkingSegment,
    ToolCallChunk,
    ToolCallSegment,
    ToolResultSegment,
)
from lib.llm.adapters.anthropic import build_messages, build_tools, decode_event


def test_build_messages_wraps_text_segment_in_text_block() -> None:
    messages = (Message(role='user', segments=(TextSegment(text='hi there'),)),)

    assert list(build_messages(messages)) == [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'hi there'}]}
    ]


def test_build_messages_parses_tool_call_arguments_into_input() -> None:
    messages = (
        Message(
            role='assistant',
            segments=(
                ToolCallSegment(id='call_1', name='search', arguments='{"q": "x"}'),
            ),
        ),
    )

    assert list(build_messages(messages)) == [
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'tool_use',
                    'id': 'call_1',
                    'name': 'search',
                    'input': {'q': 'x'},
                }
            ],
        }
    ]


def test_build_messages_remaps_tool_role_to_user_with_tool_result_block() -> None:
    messages = (
        Message(
            role='tool',
            segments=(ToolResultSegment(tool_call_id='call_1', content='42'),),
        ),
    )

    assert list(build_messages(messages)) == [
        {
            'role': 'user',
            'content': [
                {'type': 'tool_result', 'tool_use_id': 'call_1', 'content': '42'}
            ],
        }
    ]


def test_build_messages_marks_failed_tool_result_as_error() -> None:
    messages = (
        Message(
            role='tool',
            segments=(
                ToolResultSegment(tool_call_id='call_1', content='boom', error=True),
            ),
        ),
    )

    assert list(build_messages(messages)) == [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': 'call_1',
                    'content': 'boom',
                    'is_error': True,
                }
            ],
        }
    ]


def test_build_messages_roundtrips_opaque_thinking_block() -> None:
    data = {'type': 'thinking', 'thinking': 'reasoning...', 'signature': 'sig'}
    messages = (Message(role='assistant', segments=(ThinkingSegment(data=data),)),)

    assert list(build_messages(messages)) == [{'role': 'assistant', 'content': [data]}]


def test_build_messages_preserves_segment_order_within_message() -> None:
    messages = (
        Message(
            role='assistant',
            segments=(
                TextSegment(text='looking...'),
                ToolCallSegment(id='call_1', name='search', arguments='{}'),
            ),
        ),
    )

    assert list(build_messages(messages)) == [
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'looking...'},
                {'type': 'tool_use', 'id': 'call_1', 'name': 'search', 'input': {}},
            ],
        }
    ]


def test_build_messages_marks_last_block_cacheable() -> None:
    messages = (
        Message(
            role='user',
            segments=(TextSegment(text='a'), TextSegment(text='b')),
            cacheable=True,
        ),
    )

    assert list(build_messages(messages)) == [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'a'},
                {'type': 'text', 'text': 'b', 'cache_control': {'type': 'ephemeral'}},
            ],
        }
    ]


def test_build_tools_unwraps_openai_shaped_function_schema() -> None:
    tools = (
        {
            'type': 'function',
            'function': {
                'name': 'search',
                'description': 'search the web',
                'parameters': {'type': 'object', 'properties': {}},
            },
        },
    )

    assert list(build_tools(tools)) == [
        {
            'name': 'search',
            'description': 'search the web',
            'input_schema': {'type': 'object', 'properties': {}},
        }
    ]


def test_decode_event_yields_tool_call_chunk_on_tool_use_block_start() -> None:
    event = RawContentBlockStartEvent(
        type='content_block_start',
        index=0,
        content_block=ToolUseBlock(
            type='tool_use', id='call_1', name='search', input={}
        ),
    )

    assert list(decode_event(event)) == [
        ToolCallChunk(index=0, id='call_1', name='search', arguments='')
    ]


def test_decode_event_ignores_text_block_start() -> None:
    event = RawContentBlockStartEvent(
        type='content_block_start',
        index=0,
        content_block=TextBlock(type='text', text=''),
    )

    assert list(decode_event(event)) == []


def test_decode_event_yields_text_delta() -> None:
    event = RawContentBlockDeltaEvent(
        type='content_block_delta',
        index=0,
        delta=TextDelta(type='text_delta', text='hi'),
    )

    assert list(decode_event(event)) == [TextChunk(text='hi')]


def test_decode_event_yields_tool_call_chunk_on_input_json_delta() -> None:
    event = RawContentBlockDeltaEvent(
        type='content_block_delta',
        index=1,
        delta=InputJSONDelta(type='input_json_delta', partial_json='{"q'),
    )

    assert list(decode_event(event)) == [ToolCallChunk(index=1, arguments='{"q')]


def test_decode_event_yields_thinking_delta() -> None:
    event = RawContentBlockDeltaEvent(
        type='content_block_delta',
        index=0,
        delta=ThinkingDelta(type='thinking_delta', thinking='pondering'),
    )

    assert list(decode_event(event)) == [ThinkingChunk(text='pondering')]


def test_decode_event_maps_stop_reason_to_finish_chunk() -> None:
    event = RawMessageDeltaEvent(
        type='message_delta',
        delta=Delta(stop_reason='tool_use'),
        usage=MessageDeltaUsage(output_tokens=1),
    )

    assert list(decode_event(event)) == [FinishChunk(reason='tool_calls')]


def test_decode_event_skips_message_delta_without_stop_reason() -> None:
    event = RawMessageDeltaEvent(
        type='message_delta',
        delta=Delta(stop_reason=None),
        usage=MessageDeltaUsage(output_tokens=1),
    )

    assert list(decode_event(event)) == []


def test_decode_event_ignores_content_block_stop_and_message_stop() -> None:
    assert (
        list(decode_event(RawContentBlockStopEvent(type='content_block_stop', index=0)))
        == []
    )
    assert list(decode_event(RawMessageStopEvent(type='message_stop'))) == []
