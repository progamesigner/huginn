from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import cast

from httpx import Timeout
from openai import AsyncOpenAI, omit
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response import IncompleteDetails
from pydantic import JsonValue, SecretStr

from lib.http import HTTPClientManager
from lib.llm import (
    CompletionRequest,
    Effort,
    FinishChunk,
    ImageSegment,
    Message,
    OutputSchema,
    TextChunk,
    TextSegment,
    ThinkingChunk,
    ThinkingSegment,
    ToolCallChunk,
    ToolCallSegment,
    ToolResultSegment,
)
from lib.llm.adapters.openai import (
    OpenAIAdapter,
    build_messages,
    build_tools,
    decode_chunk,
)
from lib.llm.client import ProviderConfig

_TOOL: JsonValue = {
    'type': 'function',
    'function': {
        'name': 'search',
        'description': 'Search the web',
        'parameters': {'type': 'object', 'properties': {}},
        'strict': True,
    },
}


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
            'type': 'function_call',
            'call_id': 'call_1',
            'name': 'search',
            'arguments': '{}',
        },
    ]


def test_build_messages_places_assistant_text_before_its_tool_call() -> None:
    messages = (
        Message(
            role='assistant',
            segments=(
                TextSegment(text='I will search.'),
                ToolCallSegment(id='call_1', name='search', arguments='{}'),
            ),
        ),
    )

    assert list(build_messages(messages)) == [
        {'role': 'assistant', 'content': 'I will search.'},
        {
            'type': 'function_call',
            'call_id': 'call_1',
            'name': 'search',
            'arguments': '{}',
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
        {'type': 'function_call_output', 'call_id': 'call_1', 'output': '42'},
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
                {'type': 'input_text', 'text': 'what is this'},
                {
                    'type': 'input_image',
                    'detail': 'auto',
                    'image_url': 'data:image/jpeg;base64,YQ==',
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
                    'type': 'input_image',
                    'detail': 'auto',
                    'image_url': 'data:image/png;base64,Yg==',
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

    built = cast(list[dict[str, object]], list(build_messages(messages)))

    assert [entry['type'] if 'type' in entry else entry['role'] for entry in built] == [
        'user',
        'function_call',
        'function_call_output',
    ]


def test_decode_chunk_yields_text_delta() -> None:
    event = ResponseTextDeltaEvent.model_construct(
        type='response.output_text.delta',
        delta='hi',
    )

    assert list(decode_chunk(event)) == [TextChunk(text='hi')]


def test_decode_chunk_yields_reasoning_delta() -> None:
    event = ResponseReasoningTextDeltaEvent.model_construct(
        type='response.reasoning_text.delta',
        delta='considering',
    )

    assert list(decode_chunk(event)) == [ThinkingChunk(text='considering')]


def test_decode_chunk_starts_a_tool_call_from_an_output_item() -> None:
    item = ResponseFunctionToolCall.model_construct(
        type='function_call',
        call_id='call_1',
        name='search',
        arguments='',
    )
    event = ResponseOutputItemAddedEvent.model_construct(
        type='response.output_item.added',
        output_index=0,
        item=item,
    )

    assert list(decode_chunk(event)) == [
        ToolCallChunk(index=0, id='call_1', name='search'),
    ]


def test_decode_chunk_yields_a_tool_call_argument_fragment() -> None:
    event = ResponseFunctionCallArgumentsDeltaEvent.model_construct(
        type='response.function_call_arguments.delta',
        output_index=1,
        delta='{"q',
    )

    assert list(decode_chunk(event)) == [
        ToolCallChunk(index=1, arguments='{"q'),
    ]


def test_decode_chunk_maps_completed_text_to_stop() -> None:
    response = Response.model_construct(output=[])
    event = ResponseCompletedEvent.model_construct(response=response)

    assert list(decode_chunk(event)) == [FinishChunk(reason='stop')]


def test_decode_chunk_maps_completed_function_call_to_tool_calls() -> None:
    call = ResponseFunctionToolCall.model_construct(
        type='function_call',
        call_id='call_1',
        name='search',
        arguments='{}',
    )
    response = Response.model_construct(output=[call])
    event = ResponseCompletedEvent.model_construct(response=response)

    assert list(decode_chunk(event)) == [FinishChunk(reason='tool_calls')]


def test_decode_chunk_maps_max_output_tokens_to_length() -> None:
    response = Response.model_construct(
        output=[],
        incomplete_details=IncompleteDetails(reason='max_output_tokens'),
    )
    event = ResponseIncompleteEvent.model_construct(response=response)

    assert list(decode_chunk(event)) == [FinishChunk(reason='length')]


def test_decode_chunk_maps_content_filter_to_error() -> None:
    response = Response.model_construct(
        output=[],
        incomplete_details=IncompleteDetails(reason='content_filter'),
    )
    event = ResponseIncompleteEvent.model_construct(response=response)

    assert list(decode_chunk(event)) == [FinishChunk(reason='error')]


def test_decode_chunk_maps_failed_response_to_error() -> None:
    event = ResponseFailedEvent.model_construct()

    assert list(decode_chunk(event)) == [FinishChunk(reason='error')]


def test_decode_chunk_maps_stream_error_to_error() -> None:
    event = ResponseErrorEvent.model_construct()

    assert list(decode_chunk(event)) == [FinishChunk(reason='error')]


def test_build_tools_flattens_chat_completion_function_tools() -> None:
    assert list(build_tools((_TOOL,))) == [
        {
            'type': 'function',
            'name': 'search',
            'description': 'Search the web',
            'parameters': {'type': 'object', 'properties': {}},
            'strict': True,
        },
    ]


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
    effort: Effort = 'none',
    output_schema: OutputSchema | None = None,
    tools: tuple[JsonValue, ...] = (),
) -> CompletionRequest:
    return CompletionRequest(
        model='gpt-5',
        messages=(Message(role='user', segments=(TextSegment(text='hi'),)),),
        max_completion_tokens=max_completion_tokens,
        effort=effort,
        output_schema=output_schema,
        tools=tools,
    )


def _fake_client(recorded: dict[str, object]) -> Callable[..., AsyncOpenAI]:
    async def create(**kwargs: object) -> AsyncIterator[ResponseStreamEvent]:
        recorded.clear()
        recorded.update(kwargs)

        async def stream() -> AsyncIterator[ResponseStreamEvent]:
            for event in ():
                yield event

        return stream()

    def factory(**_: object) -> SimpleNamespace:
        return SimpleNamespace(responses=SimpleNamespace(create=create))

    return cast(Callable[..., AsyncOpenAI], factory)


def _recording_client(
    recorded: dict[str, object],
    client_options: dict[str, object],
) -> Callable[..., AsyncOpenAI]:
    client = _fake_client(recorded)

    def factory(**kwargs: object) -> object:
        client_options.update(kwargs)
        return client(**kwargs)

    return cast(Callable[..., AsyncOpenAI], factory)


async def _drain(adapter: OpenAIAdapter, request: CompletionRequest) -> None:
    async for _ in adapter.stream(request):
        pass


async def test_stream_disables_only_the_ai_client_read_timeout() -> None:
    recorded = {}
    client_options = {}
    http = HTTPClientManager()

    await _drain(
        OpenAIAdapter(
            _provider(),
            http,
            client_factory=_recording_client(recorded, client_options),
        ),
        _request(),
    )

    timeout = client_options['timeout']
    assert isinstance(timeout, Timeout)
    assert timeout.connect == http.timeout.connect
    assert timeout.read is None
    assert timeout.write == http.timeout.write
    assert timeout.pool == http.timeout.pool
    assert http.timeout.read == 30.0


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

    assert recorded['max_output_tokens'] == 64


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

    assert recorded['max_output_tokens'] is omit


async def test_stream_forwards_reasoning_effort_with_tools() -> None:
    recorded = {}

    await _drain(
        OpenAIAdapter(
            _provider(),
            HTTPClientManager(),
            client_factory=_fake_client(recorded),
        ),
        _request(effort='high', tools=(_TOOL,)),
    )

    assert recorded['reasoning'] == {'effort': 'high'}
    assert recorded['tools'] == list(build_tools((_TOOL,)))


async def test_stream_omits_text_format_when_no_schema() -> None:
    recorded = {}

    await _drain(
        OpenAIAdapter(
            _provider(),
            HTTPClientManager(),
            client_factory=_fake_client(recorded),
        ),
        _request(),
    )

    assert recorded['text'] is omit


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

    assert recorded['text'] == {
        'format': {
            'type': 'json_schema',
            'name': 'summary_output',
            'schema': {'type': 'object', 'properties': {}},
            'strict': True,
        },
    }
