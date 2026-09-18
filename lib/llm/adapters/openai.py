from collections.abc import AsyncIterator, Callable, Iterator
from typing import cast

from httpx import Timeout
from openai import AsyncOpenAI, omit
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseIncompleteEvent,
    ResponseInputFileParam,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputTextParam,
    ResponseOutputItemAddedEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextConfigParam,
    ResponseTextDeltaEvent,
)
from pydantic import JsonValue

from lib.http import HTTPClientContract
from lib.llm.client import ProviderConfig
from lib.llm.types import (
    CompletionChunk,
    CompletionRequest,
    DocumentSegment,
    FinishChunk,
    ImageSegment,
    Message,
    TextChunk,
    TextSegment,
    ThinkingChunk,
    ToolCallChunk,
    ToolCallSegment,
    ToolResultSegment,
)

_TOOL_ITEMS = ('function_call', 'function_call_output')


def _ephemeral() -> dict[str, str]:
    return {'type': 'ephemeral'}


def _mark(entry: ResponseInputItemParam) -> ResponseInputItemParam:
    item = dict(cast(dict[str, object], entry))

    if item.get('type') in _TOOL_ITEMS:
        item['cache_control'] = _ephemeral()
        return cast(ResponseInputItemParam, item)

    content = item.get('content')
    blocks = (
        [cast(object, ResponseInputTextParam(type='input_text', text=content))]
        if isinstance(content, str)
        else list(cast(list[object], content or []))
    )
    if blocks:
        last = dict(cast(dict[str, object], blocks[-1]))
        last['cache_control'] = _ephemeral()
        blocks[-1] = last
        item['content'] = blocks

    return cast(ResponseInputItemParam, item)


def build_messages(
    messages: tuple[Message, ...],
    cache_control: bool = False,
) -> Iterator[ResponseInputItemParam]:
    def build_entries(message: Message) -> Iterator[ResponseInputItemParam]:
        if message.role == 'tool':
            for segment in message.segments:
                match segment:
                    case ToolResultSegment(
                        tool_call_id=tool_call_id,
                        content=content,
                    ):
                        yield cast(
                            ResponseInputItemParam,
                            {
                                'type': 'function_call_output',
                                'call_id': tool_call_id,
                                'output': content,
                            },
                        )
            return

        texts = [
            segment.text
            for segment in message.segments
            if isinstance(segment, TextSegment)
        ]
        images = (
            [
                ResponseInputImageParam(
                    type='input_image',
                    detail='auto',
                    image_url=f'data:{segment.media_type};base64,{segment.data}',
                )
                for segment in message.segments
                if isinstance(segment, ImageSegment)
            ]
            if message.role == 'user'
            else []
        )
        documents = (
            [
                ResponseInputFileParam(
                    type='input_file',
                    file_data=f'data:{segment.media_type};base64,{segment.data}',
                    filename=segment.name or 'document',
                )
                for segment in message.segments
                if isinstance(segment, DocumentSegment)
            ]
            if message.role == 'user'
            else []
        )
        if texts or images or documents:
            if images or documents:
                entry = EasyInputMessageParam(
                    role=message.role,
                    content=[
                        *(
                            ResponseInputTextParam(type='input_text', text=text)
                            for text in texts
                        ),
                        *images,
                        *documents,
                    ],
                )
            else:
                entry = EasyInputMessageParam(
                    role=message.role,
                    content='\n'.join(texts),
                )
            yield cast(ResponseInputItemParam, entry)

        if message.role == 'assistant':
            for segment in message.segments:
                match segment:
                    case ToolCallSegment() as call:
                        yield ResponseFunctionToolCallParam(
                            type='function_call',
                            call_id=call.id,
                            name=call.name,
                            arguments=call.arguments,
                        )

    for message in messages:
        entries = list(build_entries(message))
        if cache_control and message.cacheable and entries:
            entries[-1] = _mark(entries[-1])
        yield from entries


def build_tools(tools: tuple[JsonValue, ...]) -> Iterator[FunctionToolParam]:
    for tool in tools:
        entry = cast(dict[str, object], tool)
        function = cast(dict[str, object], entry['function'])
        yield FunctionToolParam(
            type='function',
            name=cast(str, function['name']),
            description=cast(str | None, function.get('description')),
            parameters=cast(dict[str, object] | None, function.get('parameters')),
            strict=cast(bool | None, function.get('strict')),
        )


def decode_chunk(
    event: ResponseStreamEvent,
    tool_called: bool = False,
) -> Iterator[CompletionChunk]:
    match event:
        case ResponseTextDeltaEvent(delta=delta):
            yield TextChunk(text=delta)
        case ResponseReasoningTextDeltaEvent(delta=delta):
            yield ThinkingChunk(text=delta)
        case ResponseOutputItemAddedEvent(
            output_index=index,
            item=ResponseFunctionToolCall(call_id=call_id, name=name),
        ):
            yield ToolCallChunk(index=index, id=call_id, name=name)
        case ResponseFunctionCallArgumentsDeltaEvent(
            output_index=index,
            delta=delta,
        ):
            yield ToolCallChunk(index=index, arguments=delta)
        case ResponseCompletedEvent(response=response):
            # NOTE: a gateway may complete without an output list
            yield FinishChunk(
                reason=(
                    'tool_calls'
                    if tool_called
                    or any(
                        isinstance(item, ResponseFunctionToolCall)
                        for item in response.output or ()
                    )
                    else 'stop'
                ),
            )
        case ResponseIncompleteEvent(response=response):
            details = response.incomplete_details
            yield FinishChunk(
                reason=(
                    'length'
                    if details is not None and details.reason == 'max_output_tokens'
                    else 'error'
                ),
            )
        case ResponseFailedEvent() | ResponseErrorEvent():
            yield FinishChunk(reason='error')


class OpenAIAdapter:
    client_factory: Callable[..., AsyncOpenAI]
    provider: ProviderConfig
    http: HTTPClientContract

    def __init__(
        self,
        provider: ProviderConfig,
        http: HTTPClientContract,
        client_factory: Callable[..., AsyncOpenAI] = AsyncOpenAI,
    ) -> None:
        self.provider = provider
        self.http = http
        self.client_factory = client_factory

    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[CompletionChunk]:
        text = omit
        if request.output_schema is not None:
            text = ResponseTextConfigParam(
                format={
                    'type': 'json_schema',
                    'name': request.output_schema.name,
                    'schema': cast(
                        dict[str, object],
                        dict(request.output_schema.json_schema),
                    ),
                    'strict': True,
                },
            )

        async with self.http.connect() as http_client:
            timeout = Timeout(http_client.timeout)
            timeout.read = None
            client = self.client_factory(
                api_key=self.provider.api_key.get_secret_value(),
                base_url=self.provider.base_url,
                http_client=http_client,
                timeout=timeout,
            )
            response = await client.responses.create(
                model=request.model,
                input=list(
                    build_messages(
                        request.messages,
                        self.provider.caches(request.model),
                    ),
                ),
                tools=list(build_tools(request.tools)) if request.tools else omit,
                reasoning=(
                    {'effort': request.effort} if request.effort != 'none' else omit
                ),
                max_output_tokens=request.max_completion_tokens or omit,
                text=text,
                stream=True,
            )

            tool_called = False
            async for event in response:
                for piece in decode_chunk(event, tool_called):
                    if isinstance(piece, ToolCallChunk):
                        tool_called = True
                    yield piece
