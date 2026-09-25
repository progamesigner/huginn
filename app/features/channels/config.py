from collections.abc import Iterator, Mapping
from dataclasses import replace
from functools import reduce
from operator import or_
from typing import Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    HttpUrl,
    RootModel,
    SecretStr,
    Tag,
    create_model,
)

from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue
from app.features.agents.config import (
    AGENTS,
    DEFAULTS,
    resolve_selection,
    validate_selection,
)
from app.features.agents.exceptions import (
    MissingModelError,
    UnknownAgentReferenceError,
)
from app.features.agents.types import RoutingPolicy, SelectionPolicy

from .exceptions import (
    ConflictingChannelAgentError,
    DuplicateChannelAgentError,
    DuplicateChannelSecretError,
    IncompleteChannelRoutingError,
    IncompleteTranscriptionError,
    UnknownChannelKindError,
)
from .registry import ChannelRegistry

type TranscriptionKind = Literal['openai']


class ChannelRouterConfig(ConfigValue):
    history_turns: Annotated[int | None, Field(ge=0)] = None
    fallback: Annotated[str, Field()]
    allowed: Annotated[tuple[str, ...], Field()] = ()
    model: Annotated[str | None, Field()] = None
    provider: Annotated[str | None, Field()] = None


class TranscriptionConfig(ConfigValue):
    kind: Annotated[TranscriptionKind | None, Field()] = None
    model: Annotated[str | None, Field()] = None
    api_key: Annotated[SecretStr | None, Field()] = None
    base_url: Annotated[HttpUrl, Field()] = HttpUrl('https://api.openai.com')


class ChannelConfig(ConfigValue):
    kind: Annotated[str, Field()]
    agent: Annotated[str | None, Field()] = None
    agents: Annotated[tuple[str, ...], Field()] = ()
    router: Annotated[ChannelRouterConfig | None, Field()] = None


class ChannelConfigList(RootModel[dict[str, ChannelConfig]]):
    pass


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
        'ChannelConfigList',
        __base__=ChannelConfigList,
        root=(dict[str, element], ...),
    )


def _describe_channels(config: HuginnConfig) -> list[str]:
    channels = config.section(CHANNELS).root
    if not channels:
        return []

    entries = ', '.join(
        f'{name} ({channel.kind}, agent={channel.agent})'
        if channel.agent is not None
        else (
            f'{name} ({channel.kind}, router, {len(channel.router.allowed)} candidates)'
            if channel.router is not None
            else f'{name} ({channel.kind})'
        )
        for name, channel in sorted(channels.items())
    )
    return [f'channels: {entries}']


def _describe_transcription(config: HuginnConfig) -> list[str]:
    transcription = config.section(TRANSCRIPTION)
    if transcription.kind is None:
        return []

    return [f'transcription: {transcription.kind} model={transcription.model}']


def _validate_transcription(config: HuginnConfig) -> None:
    transcription = config.section(TRANSCRIPTION)
    given = {
        'api_key': transcription.api_key,
        'kind': transcription.kind,
        'model': transcription.model,
    }
    missing = sorted(key for key, value in given.items() if value is None)
    if missing and len(missing) < len(given):
        raise IncompleteTranscriptionError(
            f'transcription: {", ".join(missing)} is required once the section is set',
        )


def routing_policy(channel: ChannelConfig) -> RoutingPolicy:
    return RoutingPolicy(
        agent=channel.agent,
        selection=(
            None
            if channel.router is None
            else SelectionPolicy(
                fallback=channel.router.fallback,
                history_turns=channel.router.history_turns,
                model=channel.router.model,
                provider=channel.router.provider,
                allowed=frozenset(channel.router.allowed),
            )
        ),
        allowed=frozenset(channel.agents),
    )


def _tagged(kind: str, model: Any) -> Any:
    return Annotated[model, Tag(kind)]


def _validate_agent_reference(
    agents: Mapping[str, object],
    path: str,
    agent_name: str,
) -> None:
    if agent_name not in agents:
        raise UnknownAgentReferenceError(f'{path}: unknown agent {agent_name!r}')


def _duplicate_name(values: tuple[str, ...]) -> str | None:
    seen = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _validate_agent_names(
    agents: Mapping[str, object],
    values: tuple[str, ...],
    path: str,
) -> None:
    if (duplicate := _duplicate_name(values)) is not None:
        raise DuplicateChannelAgentError(f'{path}: duplicate agent {duplicate!r}')
    for index, agent_name in enumerate(values):
        _validate_agent_reference(agents, f'{path}[{index}]', agent_name)


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

        if channel.agent is not None and channel.router is not None:
            raise ConflictingChannelAgentError(
                f'{base}: agent and router cannot both be configured',
            )
        if channel.agent is None and channel.router is None:
            raise IncompleteChannelRoutingError(
                f'{base}: exactly one of agent or router is required',
            )

        _validate_agent_names(agents, channel.agents, f'{base}.agents')
        if channel.agent is not None:
            _validate_agent_reference(agents, f'{base}.agent', channel.agent)
            if channel.agents and channel.agent not in channel.agents:
                raise UnknownAgentReferenceError(
                    f'{base}.agent: {channel.agent!r} is not in {base}.agents',
                )

        if channel.router is not None:
            router = f'{base}.router'
            if not channel.router.allowed:
                raise IncompleteChannelRoutingError(
                    f'{router}.allowed: at least one candidate is required',
                )
            _validate_agent_names(agents, channel.router.allowed, f'{router}.allowed')
            if channel.agents and not set(channel.router.allowed).issubset(
                channel.agents,
            ):
                raise UnknownAgentReferenceError(
                    f'{router}.allowed: candidates must be in {base}.agents',
                )
            _validate_agent_reference(
                agents,
                f'{router}.fallback',
                channel.router.fallback,
            )
            if channel.router.fallback not in channel.router.allowed:
                raise UnknownAgentReferenceError(
                    f'{router}.fallback: {channel.router.fallback!r} is not in {router}.allowed',
                )
            try:
                ref = resolve_selection(
                    channel.router.provider,
                    channel.router.model,
                    config.section(DEFAULTS),
                )
            except MissingModelError as error:
                raise MissingModelError(f'{router}.model: {error}') from error

            validate_selection(config, router, ref)


def build_sections(registry: ChannelRegistry) -> Iterator[SectionSpec[BaseModel]]:
    yield replace(
        CHANNELS,
        thunk=lambda: _build_channels_config(registry),
        validate=lambda config: _validate_channels(config, registry),
    )
    yield TRANSCRIPTION


CHANNELS = SectionSpec(
    'channels',
    ChannelConfigList,
    describe=_describe_channels,
)

TRANSCRIPTION = SectionSpec(
    'transcription',
    TranscriptionConfig,
    describe=_describe_transcription,
    validate=_validate_transcription,
)
