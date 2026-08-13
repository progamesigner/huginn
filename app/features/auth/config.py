from collections.abc import Iterator
from typing import Annotated

from pydantic import BaseModel, Field, RootModel, SecretStr

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

from .exceptions import (
    ConflictingApiKeyAgentError,
    DuplicateApiKeyAgentError,
    DuplicateApiKeyTokenError,
    EmptyApiKeyTokenError,
    IncompleteApiKeyRoutingError,
    MissingApiKeyAgentsError,
)


class ApiKeyRouterConfig(ConfigValue):
    fallback: Annotated[str, Field()]
    allowed: Annotated[tuple[str, ...], Field()] = ()
    model: Annotated[str | None, Field()] = None
    provider: Annotated[str | None, Field()] = None


class ApiKeyToolsConfig(ConfigValue):
    deny: Annotated[tuple[str, ...], Field()] = ()


class ApiKeyConfig(ConfigValue):
    token: Annotated[SecretStr, Field()]
    agents: Annotated[tuple[str, ...], Field()] = ()
    agent: Annotated[str | None, Field()] = None
    router: Annotated[ApiKeyRouterConfig | None, Field()] = None
    tools: Annotated[ApiKeyToolsConfig | None, Field()] = None


class ApiKeyConfigList(RootModel[dict[str, ApiKeyConfig]]):
    pass


def _describe_api_keys(config: HuginnConfig) -> list[str]:
    keys = config.section(API_KEYS).root
    if not keys:
        return []

    entries = ', '.join(
        f'{name} ({", ".join(key.agents)})' for name, key in sorted(keys.items())
    )
    return [f'api_keys: {entries}']


def _validate_agent_reference(config: HuginnConfig, path: str, agent: str) -> None:
    if agent not in config.section(AGENTS).root:
        raise UnknownAgentReferenceError(f'{path}: unknown agent {agent!r}')


def _duplicate_name(values: tuple[str, ...]) -> str | None:
    seen = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _validate_agent_names(
    config: HuginnConfig,
    values: tuple[str, ...],
    path: str,
) -> None:
    if (duplicate := _duplicate_name(values)) is not None:
        raise DuplicateApiKeyAgentError(f'{path}: duplicate agent {duplicate!r}')
    for index, agent_name in enumerate(values):
        _validate_agent_reference(config, f'{path}[{index}]', agent_name)


def _validate_api_keys(config: HuginnConfig) -> None:
    tokens = {}

    for name, key in config.section(API_KEYS).root.items():
        base = f'api_keys.{name}'
        if not key.token.get_secret_value():
            raise EmptyApiKeyTokenError(f'{base}.token: must not be empty')

        owner = tokens.setdefault(key.token.get_secret_value(), name)
        if owner != name:
            raise DuplicateApiKeyTokenError(
                f'{base}.token: duplicates api_keys.{owner}',
            )

        if not key.agents:
            raise MissingApiKeyAgentsError(
                f'{base}.agents: at least one agent is required',
            )
        _validate_agent_names(config, key.agents, f'{base}.agents')

        if key.agent is not None and key.router is not None:
            raise ConflictingApiKeyAgentError(
                f'{base}: agent and router cannot both be configured',
            )
        if key.agent is not None:
            _validate_allowed(key, base, 'agent', key.agent)
        if key.router is not None:
            router = f'{base}.router'
            if not key.router.allowed:
                raise IncompleteApiKeyRoutingError(
                    f'{router}.allowed: at least one candidate is required',
                )
            _validate_agent_names(config, key.router.allowed, f'{router}.allowed')
            if not set(key.router.allowed).issubset(key.agents):
                raise UnknownAgentReferenceError(
                    f'{router}.allowed: candidates must be in {base}.agents',
                )
            _validate_allowed(key, base, 'router.fallback', key.router.fallback)
            if key.router.fallback not in key.router.allowed:
                raise UnknownAgentReferenceError(
                    f'{router}.fallback: {key.router.fallback!r} is not in {router}.allowed',
                )
            try:
                ref = resolve_selection(
                    key.router.provider,
                    key.router.model,
                    config.section(DEFAULTS),
                )
            except MissingModelError as error:
                raise MissingModelError(f'{router}.model: {error}') from error

            validate_selection(config, router, ref)


def _validate_allowed(
    key: ApiKeyConfig,
    base: str,
    path: str,
    agent: str | None,
) -> None:
    if agent is not None and agent not in key.agents:
        raise UnknownAgentReferenceError(
            f'{base}.{path}: {agent!r} is not in {base}.agents',
        )


def build_sections() -> Iterator[SectionSpec[BaseModel]]:
    yield API_KEYS


API_KEYS = SectionSpec(
    'api_keys',
    ApiKeyConfigList,
    describe=_describe_api_keys,
    validate=_validate_api_keys,
)
