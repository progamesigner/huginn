from pydantic import BaseModel

from .exceptions import DuplicateChannelKindError, UnknownChannelKindError


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
