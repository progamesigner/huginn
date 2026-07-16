from collections.abc import AsyncIterator, Iterator
from typing import cast

from openai import AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolUnionParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from pydantic import JsonValue

from lib.llm.client import ProviderConfig
from lib.llm.types import (
    CompletionChunk,
    CompletionRequest,
    FinishChunk,
    FinishReason,
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

        text = list(text_fragments(message))
        tool_calls = list(tool_call_entries(message))

        if not text and not tool_calls:
            continue

        content = '\n'.join(text) or None

        match message.role:
            case 'system':
                yield ChatCompletionSystemMessageParam(
                    role=message.role,
                    content=content or '',
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
    provider: ProviderConfig
    client: AsyncOpenAI

    def __init__(self, provider: ProviderConfig) -> None:
        self.provider = provider
        self.client = AsyncOpenAI(
            api_key=provider.api_key.get_secret_value(),
            base_url=provider.base_url,
        )

    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[CompletionChunk]:
        completion = await self.client.chat.completions.create(
            model=request.model,
            messages=build_messages(request.messages),
            tools=build_tools(request.tools) if request.tools else omit,
            reasoning_effort=request.effort if request.effort != 'none' else omit,
            stream=True,
        )

        async for chunk in completion:
            for piece in decode_chunk(chunk):
                yield piece

    async def dispose(self) -> None:
        await self.client.close()
