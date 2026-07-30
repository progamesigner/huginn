from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from app.ports.channels import InboundCommand, InboundMessage

from .exceptions import (
    DuplicateChannelKindError,
    DuplicateCommandError,
    UnknownChannelKindError,
)

type CommandHandler = Callable[[InboundCommand, InboundMessage], Awaitable[str]]


def _unknown(name: str, known: tuple[str, ...]) -> str:
    if not known:
        return f'/{name} is not a command I know.'

    available = ', '.join(f'/{command}' for command in known)

    return f'/{name} is not a command I know. Available: {available}.'


class CommandRegistry:
    handlers: dict[str, CommandHandler]

    def __init__(self) -> None:
        self.handlers = {}

    async def dispatch(self, command: InboundCommand, message: InboundMessage) -> str:
        handler = self.handlers.get(command.name)
        if handler is None:
            return _unknown(command.name, self.names())

        return await handler(command, message)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.handlers))

    def register(self, name: str, handler: CommandHandler) -> None:
        if name in self.handlers:
            raise DuplicateCommandError(f'command {name!r} is already registered')

        self.handlers[name] = handler


class ChannelRegistry:
    config_models: dict[str, type[BaseModel]]

    def __init__(self) -> None:
        self.config_models = {}

    def register(self, kind: str, config: type[BaseModel]) -> None:
        if kind in self.config_models:
            raise DuplicateChannelKindError(
                f'channel kind {kind!r} is already registered',
            )

        self.config_models[kind] = config

    def get(self, kind: str) -> type[BaseModel]:
        if kind not in self.config_models:
            available = ', '.join(sorted(self.config_models)) or 'none'
            raise UnknownChannelKindError(
                f'unknown channel kind {kind!r}; available: {available}',
            )

        return self.config_models[kind]
