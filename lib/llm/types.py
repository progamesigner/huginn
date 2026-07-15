from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

type Role = Literal['system', 'user', 'assistant', 'tool']
type Effort = Literal['none', 'low', 'medium', 'high']
type FinishReason = Literal['stop', 'tool_calls', 'length', 'error']


class BaseValue(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)


class TextSegment(BaseValue):
    text: Annotated[str, Field()]


class ToolCallSegment(BaseValue):
    id: Annotated[str, Field()]
    name: Annotated[str, Field()]
    arguments: Annotated[str, Field()]


class ToolResultSegment(BaseValue):
    tool_call_id: Annotated[str, Field()]
    content: Annotated[str, Field()]
    error: Annotated[bool, Field()] = False


class ThinkingSegment(BaseValue):
    data: Annotated[JsonValue, Field()]


type Segment = TextSegment | ToolCallSegment | ToolResultSegment | ThinkingSegment


class Message(BaseValue):
    role: Annotated[Role, Field()]
    segments: Annotated[tuple[Segment, ...], Field()]
    cacheable: Annotated[bool, Field()] = False


class TextChunk(BaseValue):
    text: Annotated[str, Field()]


class ToolCallChunk(BaseValue):
    index: Annotated[int, Field()]
    id: Annotated[str | None, Field()] = None
    name: Annotated[str | None, Field()] = None
    arguments: Annotated[str, Field()] = ''


class ThinkingChunk(BaseValue):
    text: Annotated[str, Field()]


class FinishChunk(BaseValue):
    reason: Annotated[FinishReason, Field()]


type CompletionChunk = TextChunk | ToolCallChunk | ThinkingChunk | FinishChunk


class CompletionRequest(BaseValue):
    model: Annotated[str, Field()]
    messages: Annotated[tuple[Message, ...], Field()]
    tools: Annotated[tuple[JsonValue, ...], Field()] = ()
    effort: Annotated[Effort, Field()] = 'none'
