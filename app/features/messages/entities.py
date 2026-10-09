from typing import Annotated

from pydantic import Field

from app.contracts.channels import ArtifactKind, ToolPhase, TurnEnding
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


# NOTE: a reference, never the bytes -- the reader fetches those with its own request
class AttachmentEvent(AppValue):
    blob: Annotated[str, Field()]
    kind: Annotated[ArtifactKind, Field()]
    mime: Annotated[str, Field()]
    size: Annotated[int, Field()]
    url: Annotated[str, Field()]
    name: Annotated[str | None, Field()] = None


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
    | AttachmentEvent
    | UnfinishedEvent
    | DoneEvent
    | ErrorEvent
    | PingEvent
)
