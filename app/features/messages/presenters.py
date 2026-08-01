import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import override

from pydantic import JsonValue

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

_DONE = '[DONE]'
_KEEPALIVE = 'ping'
_TURN_FAILED = 'turn_failed'


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


class OpenAIPresenter(Presenter):
    id: str
    created: int
    model: str
    opened: bool

    def __init__(self, id_: str, created: int, model: str) -> None:
        self.id = id_
        self.created = created
        self.model = model
        self.opened = False

    @override
    def present(self, event: TurnEvent) -> Iterator[ServerSentEvent]:
        match event:
            case TextEvent(text=text):
                yield self._chunk(self._delta(text))
            case DoneEvent():
                yield self._chunk({}, finish='stop')
                yield ServerSentEvent(raw_data=_DONE)
            case ErrorEvent(message=message):
                yield ServerSentEvent(
                    data={'error': {'message': message, 'type': _TURN_FAILED}},
                )
                yield ServerSentEvent(raw_data=_DONE)
            case PingEvent():
                yield ServerSentEvent(comment=_KEEPALIVE)
            case OpenedEvent() | RoutedEvent() | ToolEvent():
                return

    def _chunk(
        self,
        delta: dict[str, JsonValue],
        finish: str | None = None,
    ) -> ServerSentEvent:
        return ServerSentEvent(
            data={
                'id': self.id,
                'object': 'chat.completion.chunk',
                'created': self.created,
                'model': self.model,
                'choices': [
                    {'index': 0, 'delta': delta, 'finish_reason': finish},
                ],
            },
        )

    def _delta(self, text: str) -> dict[str, JsonValue]:
        if self.opened:
            return {'content': text}

        self.opened = True
        return {'role': 'assistant', 'content': text}


class AnthropicPresenter(Presenter):
    id: str
    model: str
    opened: bool

    def __init__(self, id_: str, model: str) -> None:
        self.id = id_
        self.model = model
        self.opened = False

    @override
    def present(self, event: TurnEvent) -> Iterator[ServerSentEvent]:
        match event:
            case TextEvent(text=text):
                yield from self._open()
                yield ServerSentEvent(
                    event='content_block_delta',
                    data={
                        'type': 'content_block_delta',
                        'index': 0,
                        'delta': {'type': 'text_delta', 'text': text},
                    },
                )
            case DoneEvent():
                yield from self._open()
                yield ServerSentEvent(
                    event='content_block_stop',
                    data={'type': 'content_block_stop', 'index': 0},
                )
                yield ServerSentEvent(
                    event='message_delta',
                    data={
                        'type': 'message_delta',
                        'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
                    },
                )
                yield ServerSentEvent(
                    event='message_stop',
                    data={'type': 'message_stop'},
                )
            case ErrorEvent(message=message):
                yield ServerSentEvent(
                    event='error',
                    data={
                        'type': 'error',
                        'error': {'type': 'api_error', 'message': message},
                    },
                )
            case PingEvent():
                yield ServerSentEvent(event='ping', data={'type': 'ping'})
            case OpenedEvent() | RoutedEvent() | ToolEvent():
                return

    def _open(self) -> Iterator[ServerSentEvent]:
        if self.opened:
            return

        self.opened = True
        yield ServerSentEvent(
            event='message_start',
            data={
                'type': 'message_start',
                'message': {
                    'id': self.id,
                    'type': 'message',
                    'role': 'assistant',
                    'model': self.model,
                    'content': [],
                    'stop_reason': None,
                },
            },
        )
        yield ServerSentEvent(
            event='content_block_start',
            data={
                'type': 'content_block_start',
                'index': 0,
                'content_block': {'type': 'text', 'text': ''},
            },
        )
