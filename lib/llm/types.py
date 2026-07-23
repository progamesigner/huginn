from collections.abc import Mapping
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

type Role = Literal['system', 'user', 'assistant', 'tool']
type Effort = Literal['none', 'low', 'medium', 'high']
type FinishReason = Literal['stop', 'tool_calls', 'length', 'error']


class LLMValue(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)


class TextSegment(LLMValue):
    text: Annotated[str, Field()]


class ToolCallSegment(LLMValue):
    id: Annotated[str, Field()]
    name: Annotated[str, Field()]
    arguments: Annotated[str, Field()]


class ToolResultSegment(LLMValue):
    tool_call_id: Annotated[str, Field()]
    content: Annotated[str, Field()]
    error: Annotated[bool, Field()] = False


class ThinkingSegment(LLMValue):
    data: Annotated[JsonValue, Field()]


type Segment = TextSegment | ToolCallSegment | ToolResultSegment | ThinkingSegment


class Message(LLMValue):
    role: Annotated[Role, Field()]
    segments: Annotated[tuple[Segment, ...], Field()]
    cacheable: Annotated[bool, Field()] = False


class TextChunk(LLMValue):
    text: Annotated[str, Field()]


class ToolCallChunk(LLMValue):
    index: Annotated[int, Field()]
    id: Annotated[str | None, Field()] = None
    name: Annotated[str | None, Field()] = None
    arguments: Annotated[str, Field()] = ''


class ThinkingChunk(LLMValue):
    text: Annotated[str, Field()]


class FinishChunk(LLMValue):
    reason: Annotated[FinishReason, Field()]


type CompletionChunk = TextChunk | ToolCallChunk | ThinkingChunk | FinishChunk


class OutputSchema(LLMValue):
    name: Annotated[str, Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')]
    json_schema: Annotated[Mapping[str, JsonValue], Field()]

    @classmethod
    def from_model(cls, model: type[BaseModel]) -> OutputSchema:
        schema = model.model_json_schema()
        name = cast(str, schema.pop('title', model.__name__))[:64]
        if 'properties' in schema:
            # NOTE: OpenAI's strict structured-output mode requires every
            #       top-level property to appear in `required`
            schema['required'] = list(cast(dict[str, object], schema['properties']))
        return cls(name=name, json_schema=schema)


class CompletionRequest(LLMValue):
    model: Annotated[str, Field()]
    messages: Annotated[tuple[Message, ...], Field()]
    tools: Annotated[tuple[JsonValue, ...], Field()] = ()
    effort: Annotated[Effort, Field()] = 'none'
    max_completion_tokens: Annotated[int | None, Field()] = None
    output_schema: Annotated[OutputSchema | None, Field()] = None
