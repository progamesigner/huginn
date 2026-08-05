from collections.abc import AsyncIterator, Callable, Iterator
from typing import cast

from httpx import Timeout
from openai import AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolUnionParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema
from pydantic import JsonValue

from lib.http import HTTPClientContract
from lib.llm.client import ProviderConfig
from lib.llm.types import (
    CompletionChunk,
    CompletionRequest,
    FinishChunk,
    FinishReason,
    ImageSegment,
    Message,
    TextChunk,
    TextSegment,
    ToolCallChunk,
    ToolCallSegment,
    ToolResultSegment,
)

_FINISH_REASONS: dict[str, FinishReason] = {
    'stop': 'stop',
    'length': 'length',
    'tool_calls': 'tool_calls',
    'function_call': 'tool_calls',
    'content_filter': 'error',
}


def build_messages(
    messages: tuple[Message, ...],
) -> Iterator[ChatCompletionMessageParam]:
    def content_parts(
        text: str | None,
        images: list[ChatCompletionContentPartImageParam],
    ) -> Iterator[ChatCompletionContentPartParam]:
        if text is not None:
            yield ChatCompletionContentPartTextParam(type='text', text=text)
        yield from images

    def image_entries(
        message: Message,
    ) -> Iterator[ChatCompletionContentPartImageParam]:
        for segment in message.segments:
            match segment:
                case ImageSegment(media_type=media_type, data=data):
                    yield ChatCompletionContentPartImageParam(
                        type='image_url',
                        image_url=ImageURL(url=f'data:{media_type};base64,{data}'),
                    )

    def text_fragments(message: Message) -> Iterator[str]:
        for segment in message.segments:
            match segment:
                case TextSegment(text=fragment):
                    yield fragment

    def tool_call_entries(
        message: Message,
    ) -> Iterator[ChatCompletionMessageFunctionToolCallParam]:
        for segment in message.segments:
            match segment:
                case ToolCallSegment(id=id_, name=name, arguments=arguments):
                    yield ChatCompletionMessageFunctionToolCallParam(
                        id=id_,
                        type='function',
                        function=Function(name=name, arguments=arguments),
                    )

    for message in messages:
        if message.role == 'tool':
            match message.segments[0]:
                case ToolResultSegment(tool_call_id=tool_call_id, content=content):
                    yield ChatCompletionToolMessageParam(
                        role='tool',
                        tool_call_id=tool_call_id,
                        content=content,
                    )
            continue

        texts = list(text_fragments(message))
        tool_calls = list(tool_call_entries(message))
        # NOTE: only a user turn can hold an image
        images = list(image_entries(message)) if message.role == 'user' else []

        if not texts and not tool_calls and not images:
            continue

        content = '\n'.join(texts) or None
        match message.role:
            case 'system':
                yield ChatCompletionSystemMessageParam(
                    role=message.role,
                    content=content or '',
                )
            case 'user' if images:
                yield ChatCompletionUserMessageParam(
                    role=message.role,
                    content=list(content_parts(content, images)),
                )
            case 'user':
                yield ChatCompletionUserMessageParam(
                    role=message.role,
                    content=content or '',
                )
            case 'assistant':
                entry = ChatCompletionAssistantMessageParam(
                    role=message.role,
                    content=content,
                )
                if tool_calls:
                    entry['tool_calls'] = tool_calls
                yield entry


def build_tools(tools: tuple[JsonValue, ...]) -> Iterator[ChatCompletionToolUnionParam]:
    for tool in tools:
        yield cast(ChatCompletionFunctionToolParam, tool)


def decode_chunk(chunk: ChatCompletionChunk) -> Iterator[CompletionChunk]:
    if not chunk.choices:
        return

    choice = chunk.choices[0]
    delta = choice.delta

    if delta.content:
        yield TextChunk(text=delta.content)

    for call in delta.tool_calls or ():
        function = call.function
        yield ToolCallChunk(
            index=call.index,
            id=call.id,
            name=function.name if function else None,
            arguments=(function.arguments if function else None) or '',
        )

    if choice.finish_reason is not None:
        yield FinishChunk(reason=_FINISH_REASONS[choice.finish_reason])


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
        response_format = omit
        if request.output_schema is not None:
            response_format = ResponseFormatJSONSchema(
                type='json_schema',
                json_schema=JSONSchema(
                    name=request.output_schema.name,
                    schema=cast(
                        dict[str, object],
                        dict(request.output_schema.json_schema),
                    ),
                    strict=True,
                ),
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
            completion = await client.chat.completions.create(
                model=request.model,
                messages=build_messages(request.messages),
                tools=build_tools(request.tools) if request.tools else omit,
                reasoning_effort=request.effort if request.effort != 'none' else omit,
                max_completion_tokens=request.max_completion_tokens or omit,
                response_format=response_format,
                stream=True,
            )

            async for chunk in completion:
                for piece in decode_chunk(chunk):
                    yield piece
