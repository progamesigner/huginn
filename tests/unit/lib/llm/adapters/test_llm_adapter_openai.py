from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import cast

from openai import AsyncOpenAI, omit
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pydantic import SecretStr

from lib.http import HTTPClientManager
from lib.llm import (
    CompletionRequest,
    FinishChunk,
    ImageSegment,
    Message,
    OutputSchema,
    TextChunk,
    TextSegment,
    ThinkingSegment,
    ToolCallChunk,
    ToolCallSegment,
    ToolResultSegment,
)
from lib.llm.adapters.openai import OpenAIAdapter, build_messages, decode_chunk
from lib.llm.client import ProviderConfig


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
                },
            ],
        },
    ]


def test_build_messages_maps_tool_result_to_tool_role_entry() -> None:
    messages = (
        Message(
            role='tool',
            segments=(ToolResultSegment(tool_call_id='call_1', content='42'),),
        ),
    )

    assert list(build_messages(messages)) == [
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': '42'},
    ]


def test_build_messages_drops_thinking_segments() -> None:
    messages = (
        Message(
            role='assistant',
            segments=(TextSegment(text='ok'), ThinkingSegment(data={'thinking': 'x'})),
        ),
    )

    assert list(build_messages(messages)) == [{'role': 'assistant', 'content': 'ok'}]


def test_build_messages_turns_an_image_into_content_parts() -> None:
    messages = (
        Message(
            role='user',
            segments=(
                TextSegment(text='what is this'),
                ImageSegment(media_type='image/jpeg', data='YQ=='),
            ),
        ),
    )

    assert list(build_messages(messages)) == [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'what is this'},
                {
                    'type': 'image_url',
                    'image_url': {'url': 'data:image/jpeg;base64,YQ=='},
                },
            ],
        },
    ]


def test_build_messages_sends_an_image_without_a_caption() -> None:
    messages = (
        Message(
            role='user',
            segments=(ImageSegment(media_type='image/png', data='Yg=='),),
        ),
    )

    assert list(build_messages(messages)) == [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image_url',
                    'image_url': {'url': 'data:image/png;base64,Yg=='},
                },
            ],
        },
    ]


# NOTE: the wire shape of a text-only turn must not change just because images exist
def test_build_messages_keeps_plain_content_for_a_text_only_turn() -> None:
    messages = (Message(role='user', segments=(TextSegment(text='hi'),)),)

    assert list(build_messages(messages)) == [{'role': 'user', 'content': 'hi'}]


def test_build_messages_drops_an_image_outside_a_user_turn() -> None:
    messages = (
        Message(
            role='system',
            segments=(
                TextSegment(text='be brief'),
                ImageSegment(media_type='image/jpeg', data='YQ=='),
            ),
        ),
    )

    assert list(build_messages(messages)) == [
        {'role': 'system', 'content': 'be brief'},
    ]


def test_build_messages_skips_a_system_message_holding_only_an_image() -> None:
    messages = (
        Message(
            role='system',
            segments=(ImageSegment(media_type='image/jpeg', data='YQ=='),),
        ),
    )

    assert list(build_messages(messages)) == []


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
        Choice(index=0, delta=ChoiceDelta(content='hi'), finish_reason=None),
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
            ),
        ],
    )
    chunk = make_chunk(Choice(index=0, delta=delta, finish_reason=None))

    assert list(decode_chunk(chunk)) == [
        ToolCallChunk(index=0, id='call_1', name='search', arguments='{"q'),
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
        ],
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
        Choice(index=0, delta=ChoiceDelta(), finish_reason='function_call'),
    )

    assert list(decode_chunk(chunk)) == [FinishChunk(reason='tool_calls')]


def test_decode_chunk_maps_content_filter_finish_reason_to_error() -> None:
    chunk = make_chunk(
        Choice(index=0, delta=ChoiceDelta(), finish_reason='content_filter'),
    )

    assert list(decode_chunk(chunk)) == [FinishChunk(reason='error')]


def _provider() -> ProviderConfig:
    return ProviderConfig(
        name='openai',
        adapter='openai',
        api_key=SecretStr('secret'),
        models=('gpt-5',),
        base_url='https://api.example',
    )


def _request(
    max_completion_tokens: int | None = None,
    output_schema: OutputSchema | None = None,
) -> CompletionRequest:
    return CompletionRequest(
        model='gpt-5',
        messages=(Message(role='user', segments=(TextSegment(text='hi'),)),),
        max_completion_tokens=max_completion_tokens,
        output_schema=output_schema,
    )


def _fake_client(recorded: dict[str, object]) -> Callable[..., AsyncOpenAI]:
    async def create(**kwargs: object) -> AsyncIterator[ChatCompletionChunk]:
        recorded.clear()
        recorded.update(kwargs)

        async def stream() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in ():
                yield chunk

        return stream()

    def factory(**_: object) -> SimpleNamespace:
        completions = SimpleNamespace(create=create)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    return cast(Callable[..., AsyncOpenAI], factory)


async def _drain(adapter: OpenAIAdapter, request: CompletionRequest) -> None:
    async for _ in adapter.stream(request):
        pass


async def test_stream_forwards_max_completion_tokens() -> None:
    recorded = {}

    await _drain(
        OpenAIAdapter(
            _provider(),
            HTTPClientManager(),
            client_factory=_fake_client(recorded),
        ),
        _request(max_completion_tokens=64),
    )

    assert recorded['max_completion_tokens'] == 64


async def test_stream_omits_max_completion_tokens_when_unset() -> None:
    recorded = {}

    await _drain(
        OpenAIAdapter(
            _provider(),
            HTTPClientManager(),
            client_factory=_fake_client(recorded),
        ),
        _request(),
    )

    assert recorded['max_completion_tokens'] is omit


async def test_stream_omits_response_format_when_no_schema() -> None:
    recorded = {}

    await _drain(
        OpenAIAdapter(
            _provider(),
            HTTPClientManager(),
            client_factory=_fake_client(recorded),
        ),
        _request(),
    )

    assert recorded['response_format'] is omit


async def test_stream_sets_response_format_json_schema_with_name_and_schema_and_strict() -> (
    None
):
    recorded = {}
    output_schema = OutputSchema(
        name='summary_output',
        json_schema={'type': 'object', 'properties': {}},
    )

    await _drain(
        OpenAIAdapter(
            _provider(),
            HTTPClientManager(),
            client_factory=_fake_client(recorded),
        ),
        _request(output_schema=output_schema),
    )

    assert recorded['response_format'] == {
        'type': 'json_schema',
        'json_schema': {
            'name': 'summary_output',
            'schema': {'type': 'object', 'properties': {}},
            'strict': True,
        },
    }
