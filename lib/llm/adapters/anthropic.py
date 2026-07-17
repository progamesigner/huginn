from collections.abc import AsyncIterator, Iterator
from json import loads
from typing import cast

from anthropic import AsyncAnthropic, omit
from anthropic.types import (
    CacheControlEphemeralParam,
    InputJSONDelta,
    MessageParam,
    OutputConfigParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStreamEvent,
    TextBlockParam,
    TextDelta,
    ThinkingBlockParam,
    ThinkingConfigAdaptiveParam,
    ThinkingDelta,
    ToolParam,
    ToolResultBlockParam,
    ToolUnionParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from pydantic import JsonValue

from lib.http import HTTPClientManager
from lib.llm.client import ProviderConfig
from lib.llm.types import (
    CompletionChunk,
    CompletionRequest,
    FinishChunk,
    FinishReason,
    Message,
    TextChunk,
    TextSegment,
    ThinkingChunk,
    ThinkingSegment,
    ToolCallChunk,
    ToolCallSegment,
    ToolResultSegment,
)

_DEFAULT_MAX_TOKENS = 8192

_STOP_REASONS: dict[str, FinishReason] = {
    'end_turn': 'stop',
    'stop_sequence': 'stop',
    'pause_turn': 'stop',
    'max_tokens': 'length',
    'tool_use': 'tool_calls',
    'refusal': 'error',
}

type _Block = (
    TextBlockParam | ToolUseBlockParam | ToolResultBlockParam | ThinkingBlockParam
)


def build_messages(messages: tuple[Message, ...]) -> Iterator[MessageParam]:
    def build_blocks(message: Message) -> Iterator[_Block]:
        last_index = len(message.segments) - 1

        for index, segment in enumerate(message.segments):
            cache_control = (
                CacheControlEphemeralParam(type='ephemeral')
                if message.cacheable and index == last_index
                else None
            )

            match segment:
                case TextSegment(text=text):
                    block = TextBlockParam(type='text', text=text)
                    if cache_control:
                        block['cache_control'] = cache_control
                    yield block
                case ToolCallSegment(id=id_, name=name, arguments=arguments):
                    tool_use = ToolUseBlockParam(
                        type='tool_use',
                        id=id_,
                        name=name,
                        input=loads(arguments),
                    )
                    if cache_control:
                        tool_use['cache_control'] = cache_control
                    yield tool_use
                case ToolResultSegment(
                    tool_call_id=tool_call_id,
                    content=content,
                    error=error,
                ):
                    result = ToolResultBlockParam(
                        type='tool_result',
                        tool_use_id=tool_call_id,
                        content=content,
                    )
                    if error:
                        result['is_error'] = True
                    if cache_control:
                        result['cache_control'] = cache_control
                    yield result
                case ThinkingSegment(data=data):
                    fields = cast(dict[str, object], data)
                    # NOTE: thinking blocks have no cache_control field
                    yield ThinkingBlockParam(
                        type='thinking',
                        thinking=cast(str, fields['thinking']),
                        signature=cast(str, fields['signature']),
                    )

    for message in messages:
        blocks = list(build_blocks(message))
        if not blocks:
            continue

        role = 'user' if message.role == 'tool' else message.role
        yield MessageParam(role=role, content=blocks)


def build_tools(tools: tuple[JsonValue, ...]) -> Iterator[ToolUnionParam]:
    for tool in tools:
        entry = cast(dict[str, object], tool)
        function = cast(dict[str, object], entry['function'])
        yield ToolParam(
            name=cast(str, function['name']),
            description=cast(str, function.get('description', '')),
            input_schema=cast(dict[str, object], function['parameters']),
        )


def decode_event(event: RawMessageStreamEvent) -> Iterator[CompletionChunk]:
    match event:
        case RawContentBlockStartEvent(
            content_block=ToolUseBlock(id=id_, name=name),
            index=index,
        ):
            yield ToolCallChunk(index=index, id=id_, name=name, arguments='')
        case RawContentBlockDeltaEvent(delta=TextDelta(text=text)):
            yield TextChunk(text=text)
        case RawContentBlockDeltaEvent(
            delta=InputJSONDelta(partial_json=partial_json),
            index=index,
        ):
            yield ToolCallChunk(index=index, arguments=partial_json)
        case RawContentBlockDeltaEvent(delta=ThinkingDelta(thinking=thinking)):
            yield ThinkingChunk(text=thinking)
        case RawMessageDeltaEvent() if event.delta.stop_reason is not None:
            yield FinishChunk(reason=_STOP_REASONS[event.delta.stop_reason])


class AnthropicAdapter:
    provider: ProviderConfig
    http: HTTPClientManager
    max_tokens: int

    def __init__(
        self,
        provider: ProviderConfig,
        http: HTTPClientManager,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self.provider = provider
        self.http = http
        self.max_tokens = max_tokens

    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[CompletionChunk]:
        thinking = omit
        output_config = omit
        if request.effort != 'none':
            thinking = ThinkingConfigAdaptiveParam(type='adaptive')
            output_config = OutputConfigParam(effort=request.effort)

        async with self.http.connect() as http_client:
            client = AsyncAnthropic(
                api_key=self.provider.api_key.get_secret_value(),
                base_url=self.provider.base_url,
                http_client=http_client,
            )
            message = await client.messages.create(
                model=request.model,
                max_tokens=self.max_tokens,
                messages=build_messages(request.messages),
                tools=build_tools(request.tools) if request.tools else omit,
                thinking=thinking,
                output_config=output_config,
                stream=True,
            )

            async for event in message:
                for piece in decode_event(event):
                    yield piece
