import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import override

from lib.router import ServerSentEvent, format_sse_event

from .types import (
    DoneEvent,
    ErrorEvent,
    OpenedEvent,
    PingEvent,
    RoutedEvent,
    TextEvent,
    ToolEvent,
    TurnEvent,
)

_KEEPALIVE = 'ping'


def _payload(event: ServerSentEvent) -> str | None:
    if event.raw_data is not None:
        return event.raw_data

    return None if event.data is None else json.dumps(event.data, ensure_ascii=False)


def encode(event: ServerSentEvent) -> str:
    return format_sse_event(
        comment=event.comment,
        data_str=_payload(event),
        event=event.event,
    ).decode()


class Presenter(ABC):
    @abstractmethod
    def present(self, event: TurnEvent) -> Iterator[ServerSentEvent]: ...


class NativePresenter(Presenter):
    @override
    def present(self, event: TurnEvent) -> Iterator[ServerSentEvent]:
        match event:
            case OpenedEvent(conversation=conversation):
                yield ServerSentEvent(
                    event='opened',
                    data={'conversation': conversation},
                )
            case RoutedEvent(agent=agent):
                yield ServerSentEvent(event='routed', data={'agent': agent})
            case TextEvent(text=text):
                yield ServerSentEvent(event='delta', data={'text': text})
            case ToolEvent(name=name, phase=phase):
                yield ServerSentEvent(
                    event='tool',
                    data={'name': name, 'phase': phase},
                )
            case DoneEvent():
                yield ServerSentEvent(event='done', data={})
            case ErrorEvent(message=message):
                yield ServerSentEvent(event='error', data={'message': message})
            case PingEvent():
                yield ServerSentEvent(comment=_KEEPALIVE)
