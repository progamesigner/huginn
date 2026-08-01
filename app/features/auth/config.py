from collections.abc import Iterator
from typing import Annotated

from pydantic import BaseModel, Field, RootModel, SecretStr

from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue
from app.features.agents.config import AGENTS, DEFAULTS, resolve_model
from app.features.agents.exceptions import (
    MissingModelError,
    UnknownAgentReferenceError,
)

from .exceptions import (
    DuplicateApiKeyTokenError,
    EmptyApiKeyTokenError,
    MissingApiKeyAgentsError,
)


class ApiKeyRouterConfig(ConfigValue):
    fallback: Annotated[str, Field()]
    model: Annotated[str | None, Field()] = None


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

        for index, agent in enumerate(key.agents):
            _validate_agent_reference(config, f'{base}.agents[{index}]', agent)

        _validate_allowed(key, base, 'agent', key.agent)
        if key.router is not None:
            _validate_allowed(key, base, 'router.fallback', key.router.fallback)
            try:
                resolve_model(key.router.model, config.section(DEFAULTS).model)
            except MissingModelError as error:
                raise MissingModelError(f'{base}.router.model: {error}') from error


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
