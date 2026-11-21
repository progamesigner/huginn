import json
from base64 import b64decode
from collections.abc import AsyncIterator, Callable
from typing import cast
from uuid import uuid4

from google import genai
from google.genai import types
from httpx2 import Timeout
from pydantic import JsonValue

from lib.llm.client import ProviderConfig
from lib.llm.contracts import HTTPConnector
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
    ThinkingSegment,
    ToolCallChunk,
    ToolCallSegment,
    ToolResultSegment,
)


def build_contents(messages: tuple[Message, ...]) -> list[types.Content]:
    contents = []
    calls = {}
    for message in messages:
        if message.role == 'system':
            continue
        parts = []
        replay = []
        for segment in message.segments:
            match segment:
                case TextSegment(text=text):
                    if text:
                        parts.append(types.Part(text=text))
                case (
                    ImageSegment(media_type=mime, data=data)
                    | DocumentSegment(
                        media_type=mime,
                        data=data,
                    )
                ):
                    parts.append(
                        types.Part.from_bytes(data=b64decode(data), mime_type=mime),
                    )
                case ToolCallSegment(id=call_id, name=name, arguments=arguments):
                    calls[call_id] = (name, call_id)
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                id=call_id,
                                name=name,
                                args=json.loads(arguments),
                            ),
                        ),
                    )
                case ToolResultSegment(
                    tool_call_id=call_id,
                    content=content,
                    error=error,
                ):
                    if call_id not in calls:
                        raise ValueError(
                            'Gemini tool result has no matching function call',
                        )
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=calls[call_id][1],
                                name=calls[call_id][0],
                                response={'error' if error else 'output': content},
                            ),
                        ),
                    )
                case ThinkingSegment(data={'vendor': 'gemini', 'part': dict(part)}):
                    replay.append(types.Part.model_validate(part))
        # Preserve signed parts in their original order, including signature-only parts.
        if replay:
            native_calls = [part.function_call for part in replay if part.function_call]
            tool_calls = [
                segment
                for segment in message.segments
                if isinstance(segment, ToolCallSegment)
            ]
            for call, native in zip(tool_calls, native_calls, strict=True):
                calls[call.id] = (call.name, native.id)
            parts = replay
        if not parts:
            continue
        role = 'model' if message.role == 'assistant' else 'user'
        if contents and contents[-1].role == role:
            contents[-1].parts = [*(contents[-1].parts or []), *parts]
        else:
            contents.append(types.Content(role=role, parts=parts))
    return contents


def build_config(request: CompletionRequest) -> types.GenerateContentConfig:
    instructions = '\n\n'.join(
        segment.text
        for message in request.messages
        if message.role == 'system'
        for segment in message.segments
        if isinstance(segment, TextSegment)
    )
    declarations = []
    for tool in request.tools:
        entry = cast(dict[str, JsonValue], tool)
        function = cast(dict[str, JsonValue], entry['function'])
        declarations.append(
            types.FunctionDeclaration(
                name=cast(str, function['name']),
                description=cast(str, function.get('description', '')),
                parameters_json_schema=function['parameters'],
            ),
        )
    thinking = None
    if request.effort != 'none':
        if request.model.removeprefix('models/').startswith('gemini-2.5'):
            thinking = types.ThinkingConfig(
                thinking_budget={'low': 1024, 'medium': 8192, 'high': 24576}[
                    request.effort
                ],
            )
        else:
            thinking = types.ThinkingConfig(thinking_level=request.effort.upper())
    return types.GenerateContentConfig(
        system_instruction=instructions or None,
        max_output_tokens=request.max_completion_tokens,
        tools=[types.Tool(function_declarations=declarations)]
        if declarations
        else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=thinking,
        response_mime_type='application/json' if request.output_schema else None,
        response_json_schema=(
            dict(request.output_schema.json_schema) if request.output_schema else None
        ),
    )


class GeminiAdapter:
    def __init__(
        self,
        provider: ProviderConfig,
        http: HTTPConnector,
        client_factory: Callable[..., genai.Client] = genai.Client,
    ) -> None:
        self.provider = provider
        self.http = http
        self.client_factory = client_factory

    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[CompletionChunk]:
        async with self.http.connect() as http_client:
            timeout = Timeout(http_client.timeout)
            timeout.read = None
            http_client.timeout = timeout
            options = types.HttpOptions(
                api_version='v1beta',
                base_url=self.provider.base_url,
                httpx_async_client=http_client,
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            client = self.client_factory(
                enterprise=False,
                api_key=self.provider.api_key.get_secret_value(),
                http_options=options,
            )
            index = 0
            called = False
            finished = False
            try:
                events = await client.aio.models.generate_content_stream(
                    model=request.model,
                    contents=build_contents(request.messages),
                    config=build_config(request),
                )
                async for event in events:
                    if not event.candidates:
                        if event.prompt_feedback and event.prompt_feedback.block_reason:
                            yield FinishChunk(reason='error')
                            finished = True
                        continue
                    candidate = event.candidates[0]
                    for part in (
                        (candidate.content.parts or []) if candidate.content else []
                    ):
                        if part.function_call:
                            call = part.function_call
                            if not call.name:
                                raise ValueError(
                                    'Gemini returned an unnamed function call',
                                )
                            call_id = call.id or f'gemini_{uuid4().hex}'
                            yield ToolCallChunk(
                                index=index,
                                id=call_id,
                                name=call.name,
                                arguments=json.dumps(
                                    call.args or {},
                                    ensure_ascii=False,
                                ),
                            )
                            index += 1
                            called = True
                        elif part.text:
                            if part.thought:
                                yield ThinkingChunk(text=part.text)
                            else:
                                yield TextChunk(text=part.text)
                        yield ThinkingChunk(
                            text='',
                            data={
                                'vendor': 'gemini',
                                'part': cast(
                                    JsonValue,
                                    part.model_dump(mode='json', exclude_none=True),
                                ),
                            },
                        )
                    if candidate.finish_reason:
                        reason = candidate.finish_reason.value
                        yield FinishChunk(
                            reason=(
                                'length'
                                if reason == 'MAX_TOKENS'
                                else ('tool_calls' if called else 'stop')
                                if reason == 'STOP'
                                else 'error'
                            ),
                        )
                        finished = True
                if not finished:
                    raise ValueError('Gemini stream ended without a finish reason')
            finally:
                await client.aio.aclose()
                client.close()
