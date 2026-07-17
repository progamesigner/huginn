from typing import Annotated, Literal

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


class CompletionRequest(LLMValue):
    model: Annotated[str, Field()]
    messages: Annotated[tuple[Message, ...], Field()]
    tools: Annotated[tuple[JsonValue, ...], Field()] = ()
    effort: Annotated[Effort, Field()] = 'none'
