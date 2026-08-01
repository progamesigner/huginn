from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from app.contracts.channels import ToolPhase
from app.core.schema import AppRequest, AppResponse


class ChatRequest(AppRequest):
    message: Annotated[str, Field()]
    conversation: Annotated[str | None, Field()] = None
    agent: Annotated[str | None, Field()] = None


class ContentPart(AppRequest):
    type: Annotated[str, Field()] = 'text'
    text: Annotated[str, Field()] = ''


class IncomingMessage(AppRequest):
    role: Annotated[str, Field()]
    content: Annotated[str | tuple[ContentPart, ...], Field()]

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content

        return ''.join(part.text for part in self.content if part.type == 'text')


class CompatibleRequest(AppRequest):
    model: Annotated[str, Field()] = ''
    messages: Annotated[tuple[IncomingMessage, ...], Field()] = ()
    stream: Annotated[bool, Field()] = False
    conversation: Annotated[str | None, Field()] = None
    user: Annotated[str | None, Field()] = None

    def latest(self) -> str:
        return next(
            (
                message.text()
                for message in reversed(self.messages)
                if message.role == 'user'
            ),
            '',
        )

    def thread(self) -> str | None:
        return self.conversation or self.user


class CompletionMessage(AppResponse):
    role: Annotated[Literal['assistant'], Field()] = 'assistant'
    content: Annotated[str, Field()]


class CompletionChoice(AppResponse):
    index: Annotated[int, Field()] = 0
    message: Annotated[CompletionMessage, Field()]
    finish_reason: Annotated[str, Field()] = 'stop'


class Completion(AppResponse):
    id: Annotated[str, Field()]
    object: Annotated[Literal['chat.completion'], Field()] = 'chat.completion'
    created: Annotated[int, Field()]
    model: Annotated[str, Field()]
    choices: Annotated[tuple[CompletionChoice, ...], Field()]


@dataclass(frozen=True)
class OpenedEvent:
    conversation: str


@dataclass(frozen=True)
class RoutedEvent:
    agent: str


@dataclass(frozen=True)
class TextEvent:
    text: str


@dataclass(frozen=True)
class ToolEvent:
    name: str
    phase: ToolPhase


@dataclass(frozen=True)
class DoneEvent:
    pass


@dataclass(frozen=True)
class ErrorEvent:
    message: str


@dataclass(frozen=True)
class PingEvent:
    pass


type TurnEvent = (
    OpenedEvent
    | RoutedEvent
    | TextEvent
    | ToolEvent
    | DoneEvent
    | ErrorEvent
    | PingEvent
)
