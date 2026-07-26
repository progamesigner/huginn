from collections.abc import Iterator, Mapping
from dataclasses import replace
from functools import reduce
from operator import or_
from typing import Annotated, Any, cast

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    RootModel,
    SecretStr,
    Tag,
    create_model,
)

from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue
from app.features.agents.config import AGENTS, DEFAULTS, resolve_model
from app.features.agents.exceptions import (
    MissingModelError,
    UnknownAgentReferenceError,
)

from .exceptions import DuplicateChannelSecretError, UnknownChannelKindError
from .registry import ChannelRegistry


class ChannelRouterConfig(ConfigValue):
    fallback: Annotated[str, Field()]
    model: Annotated[str | None, Field()] = None


class ChannelConfig(ConfigValue):
    kind: Annotated[str, Field()]
    agent: Annotated[str | None, Field()] = None
    bindings: Annotated[dict[str, str], Field()] = Field(default_factory=dict)
    router: Annotated[ChannelRouterConfig | None, Field()] = None


class ChannelsConfig(RootModel[dict[str, ChannelConfig]]):
    pass


def _tagged(kind: str, model: Any) -> Any:
    return Annotated[model, Tag(kind)]


def _channel_config_element(configs: dict[str, type[BaseModel]]) -> Any:
    if not configs:
        return ChannelConfig

    union = reduce(or_, (_tagged(kind, model) for kind, model in configs.items()))
    return Annotated[union, Discriminator(_channel_kind)]


def _channel_kind(value: object) -> str | None:
    match value:
        case {'kind': str(kind)} | object(kind=str(kind)):
            return kind
        case _:
            return None


def _build_channels_config(registry: ChannelRegistry) -> type[BaseModel]:
    element = _channel_config_element(registry.config_models)
    return create_model(
        'ChannelsConfig',
        __base__=ChannelsConfig,
        root=(dict[str, element], ...),
    )


def _describe_channels(config: HuginnConfig) -> list[str]:
    channels = config.section(CHANNELS).root
    if not channels:
        return []

    entries = ', '.join(
        f'{name} ({channel.kind})' for name, channel in sorted(channels.items())
    )
    return [f'channels: {entries}']


def _validate_agent_reference(
    agents: Mapping[str, object],
    path: str,
    agent_name: str,
) -> None:
    if agent_name not in agents:
        raise UnknownAgentReferenceError(f'{path}: unknown agent {agent_name!r}')


def _validate_channels(config: HuginnConfig, registry: ChannelRegistry) -> None:
    agents = config.section(AGENTS).root
    secrets = {}

    for name, channel in config.section(CHANNELS).root.items():
        base = f'channels.{name}'
        try:
            registry.get(channel.kind)
        except UnknownChannelKindError as error:
            raise UnknownChannelKindError(f'{base}.kind: {error}') from error

        if (
            secret := cast(SecretStr | None, getattr(channel, 'secret_token', None))
        ) is not None:
            owner = secrets.setdefault(secret.get_secret_value(), name)
            if owner != name:
                raise DuplicateChannelSecretError(
                    f'{base}.secret_token: duplicates channels.{owner}',
                )

        if channel.agent is not None:
            _validate_agent_reference(agents, f'{base}.agent', channel.agent)

        for chat_id, agent_name in channel.bindings.items():
            _validate_agent_reference(
                agents,
                f'{base}.bindings.{chat_id}',
                agent_name,
            )

        if channel.router is not None:
            _validate_agent_reference(
                agents,
                f'{base}.router.fallback',
                channel.router.fallback,
            )
            try:
                resolve_model(channel.router.model, config.section(DEFAULTS).model)
            except MissingModelError as error:
                raise MissingModelError(f'{base}.router.model: {error}') from error


def build_sections(registry: ChannelRegistry) -> Iterator[SectionSpec[BaseModel]]:
    yield replace(
        CHANNELS,
        thunk=lambda: _build_channels_config(registry),
        validate=lambda config: _validate_channels(config, registry),
    )


CHANNELS = SectionSpec(
    'channels',
    ChannelsConfig,
    describe=_describe_channels,
)
