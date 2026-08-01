from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from app.contracts.channels import ToolPhase
from app.core.schema import AppRequest


class ChatRequest(AppRequest):
    message: Annotated[str, Field()]
    conversation: Annotated[str | None, Field()] = None
    agent: Annotated[str | None, Field()] = None


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
