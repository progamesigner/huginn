from typing import Annotated

from pydantic import Field

from app.contracts.channels import ToolPhase, TurnEnding
from app.core.schema import AppValue


class OpenedEvent(AppValue):
    conversation: Annotated[str, Field()]


class RoutedEvent(AppValue):
    agent: Annotated[str, Field()]


class TextEvent(AppValue):
    text: Annotated[str, Field()]


class ToolEvent(AppValue):
    name: Annotated[str, Field()]
    phase: Annotated[ToolPhase, Field()]


class UnfinishedEvent(AppValue):
    ending: Annotated[TurnEnding, Field()]


class DoneEvent(AppValue):
    pass


class ErrorEvent(AppValue):
    message: Annotated[str, Field()]


class PingEvent(AppValue):
    pass


type TurnEvent = (
    OpenedEvent
    | RoutedEvent
    | TextEvent
    | ToolEvent
    | UnfinishedEvent
    | DoneEvent
    | ErrorEvent
    | PingEvent
)
