from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, JsonValue, RootModel, model_validator

from app.contracts.auth import RouterEngine
from app.core.config.providers import decision_providers, llm_providers
from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue
from lib.decision import MAX_CHOICES
from lib.llm import ModelRef

from .exceptions import (
    AgentError,
    AmbiguousInstructionsError,
    AmbiguousModelError,
    MissingModelError,
    ProviderModelMismatchError,
    ReservedAgentNameError,
    RouterShadowError,
    RouterThresholdError,
    TooManyRouterCandidatesError,
    UnknownProviderReferenceError,
)

# NOTE: the answer that means no agent fits; it takes one of the choices a decision
# offers, so one fewer is left for agents
NO_AGENT = '__none__'
MAX_ROUTED_AGENTS = MAX_CHOICES - 1


class DefaultsConfig(ConfigValue):
    model: Annotated[str | None, Field()] = None
    provider: Annotated[str | None, Field()] = None
    router_prompt: Annotated[str | None, Field()] = None
    router_instructions: Annotated[str | None, Field(min_length=1)] = None
    max_completion_tokens: Annotated[int | None, Field(gt=0)] = None
    max_iterations: Annotated[int | None, Field(gt=0)] = None
    turn_timeout: Annotated[float | None, Field(gt=0)] = None


type BindingSource = Literal[
    'account',
    'actor',
    'agent',
    'channel',
    'chat',
    'handle',
    'memory_scope',
]


class InjectionConfig(ConfigValue):
    inject: Annotated[dict[str, JsonValue], Field()] = Field(default_factory=dict)
    bind: Annotated[dict[str, BindingSource], Field()] = Field(default_factory=dict)

    @model_validator(mode='after')
    def disjoint_injection(self) -> Self:
        if overlap := sorted(self.inject.keys() & self.bind.keys()):
            raise ValueError(f'inject and bind overlap: {", ".join(overlap)}')
        return self


class ToolRef(InjectionConfig):
    name: Annotated[str, Field()]
    params: Annotated[dict[str, JsonValue], Field()] = Field(default_factory=dict)


class MCPRef(InjectionConfig):
    server: Annotated[str, Field()]
    allow: Annotated[tuple[str, ...], Field()] = ()


class RouterShadowConfig(ConfigValue):
    model: Annotated[str, Field()]
    provider: Annotated[str | None, Field()] = None


class RouterConfig(ConfigValue):
    engine: Annotated[RouterEngine, Field()] = 'llm'
    history_turns: Annotated[int | None, Field(ge=0)] = None
    fallback: Annotated[str, Field()]
    allowed: Annotated[tuple[str, ...], Field()] = ()
    model: Annotated[str | None, Field()] = None
    provider: Annotated[str | None, Field()] = None
    min_confidence: Annotated[float | None, Field(ge=0, le=1)] = None
    min_margin: Annotated[float | None, Field(ge=0, le=1)] = None
    shadow: Annotated[RouterShadowConfig | None, Field()] = None

    @property
    def has_thresholds(self) -> bool:
        return self.min_confidence is not None or self.min_margin is not None


class SkillsConfig(ConfigValue):
    source: Annotated[str | None, Field(min_length=1)] = None
    enabled: Annotated[bool, Field()] = False
    allow: Annotated[tuple[str, ...], Field()] = ()


class AgentConfig(ConfigValue):
    description: Annotated[str, Field()]
    model: Annotated[str | None, Field()] = None
    provider: Annotated[str | None, Field()] = None
    max_completion_tokens: Annotated[int | None, Field(gt=0)] = None
    max_iterations: Annotated[int | None, Field(gt=0)] = None
    turn_timeout: Annotated[float | None, Field(gt=0)] = None
    tool_timeout: Annotated[float | None, Field(gt=0)] = None
    instructions: Annotated[str | None, Field()] = None
    instructions_file: Annotated[str | None, Field()] = None
    tools: Annotated[tuple[ToolRef, ...], Field()] = ()
    mcp: Annotated[tuple[MCPRef, ...], Field()] = ()
    hooks: Annotated[tuple[str, ...], Field()] = ()
    skills: Annotated[SkillsConfig, Field()] = Field(default_factory=SkillsConfig)


class AgentConfigList(RootModel[dict[str, AgentConfig]]):
    pass


def _describe_agent(name: str, agent: AgentConfig) -> str:
    model = agent.model or 'default'
    provider = agent.provider or 'any'
    tools = ', '.join(tool.name for tool in agent.tools) or 'none'
    servers = ', '.join(ref.server for ref in agent.mcp) or 'none'
    return f'  {name}: provider={provider} model={model} tools={tools} mcp={servers}'


def _describe_agents(config: HuginnConfig) -> list[str]:
    # NOTE: instructions are secrets by policy; the summary must never echo them
    agents = config.section(AGENTS).root
    lines = [f'agents ({len(agents)}):']
    lines.extend(_describe_agent(name, agent) for name, agent in agents.items())
    return lines


def _validate_agents(config: HuginnConfig) -> None:
    defaults = config.section(DEFAULTS)

    for agent_name, agent in config.section(AGENTS).root.items():
        base = f'agents.{agent_name}'

        if (agent.instructions is None) == (agent.instructions_file is None):
            raise AmbiguousInstructionsError(
                f'{base}: exactly one of instructions/instructions_file is required',
            )

        try:
            ref = resolve_selection(agent.provider, agent.model, defaults)
        except MissingModelError as error:
            raise MissingModelError(f'{base}.model: {error}') from error

        validate_selection(config, base, ref)


def build_sections() -> Iterator[SectionSpec[BaseModel]]:
    yield replace(AGENTS, validate=_validate_agents)
    yield DEFAULTS


def resolve_selection(
    provider: str | None,
    model: str | None,
    defaults: DefaultsConfig,
) -> ModelRef:
    if (resolved := model or defaults.model) is None:
        raise MissingModelError('no model configured; set the model or defaults.model')

    inherited = None if model is not None else defaults.provider
    return ModelRef(model=resolved, provider=provider or inherited)


def validate_selection(config: HuginnConfig, base: str, ref: ModelRef) -> None:
    providers = llm_providers(config)
    if ref.provider is not None:
        if (provider := providers.get(ref.provider)) is None:
            raise UnknownProviderReferenceError(
                f'{base}.provider: unknown provider {ref.provider!r}',
            )

        if ref.model not in provider.models:
            raise ProviderModelMismatchError(
                f'{base}.model: provider {ref.provider!r} does not list {ref.model!r}',
            )

        return

    owners = sorted(
        name for name, provider in providers.items() if ref.model in provider.models
    )
    if len(owners) > 1:
        raise AmbiguousModelError(
            f'{base}.model: {ref.model!r} is offered by {", ".join(owners)};'
            f' name one in {base}.provider',
        )


def validate_router_selection(
    config: HuginnConfig,
    base: str,
    *,
    engine: RouterEngine,
    provider: str | None,
    model: str | None,
    allowed: tuple[str, ...],
    thresholds: bool,
    shadow: RouterShadowConfig | None,
) -> None:
    if thresholds and engine != 'decision':
        raise RouterThresholdError(
            f'{base}: min_confidence and min_margin need engine: decision',
        )
    if shadow is not None:
        if engine != 'llm':
            raise RouterShadowError(f'{base}.shadow: only an llm router is shadowed')
        _validate_decision_candidates(config, base, allowed)
        _validate_decision_model(
            config,
            f'{base}.shadow',
            shadow.provider,
            shadow.model,
        )
    if engine == 'llm':
        try:
            ref = resolve_selection(provider, model, config.section(DEFAULTS))
        except MissingModelError as error:
            raise MissingModelError(f'{base}.model: {error}') from error
        validate_selection(config, base, ref)
        return
    if not model:
        raise MissingModelError(f'{base}.model: decision routing requires a model')
    _validate_decision_candidates(config, base, allowed)
    _validate_decision_model(config, base, provider, model)


def _validate_decision_candidates(
    config: HuginnConfig,
    base: str,
    allowed: tuple[str, ...],
) -> None:
    if len(allowed) > MAX_ROUTED_AGENTS:
        raise TooManyRouterCandidatesError(
            f'{base}.allowed: decision routing takes at most {MAX_ROUTED_AGENTS} agents',
        )
    if NO_AGENT in config.section(AGENTS).root:
        raise ReservedAgentNameError(
            f'agents.{NO_AGENT}: the name is reserved for "no agent fits"',
        )


def _validate_decision_model(
    config: HuginnConfig,
    base: str,
    provider: str | None,
    model: str,
) -> None:
    providers = decision_providers(config)
    if provider is not None:
        if provider not in providers:
            raise UnknownProviderReferenceError(
                f'{base}.provider: unknown decision provider {provider!r}',
            )
        if model not in providers[provider].models:
            raise ProviderModelMismatchError(
                f'{base}.model: not offered by decision provider {provider!r}',
            )
        return
    owners = [name for name, entry in providers.items() if model in entry.models]
    if not owners:
        raise ProviderModelMismatchError(
            f'{base}.model: no decision provider offers {model!r}',
        )
    if len(owners) > 1:
        raise AmbiguousModelError(f'{base}.provider: name one of {", ".join(owners)}')


def load_instructions(agent: AgentConfig) -> str:
    if agent.instructions is not None:
        return agent.instructions

    if agent.instructions_file is not None:
        return Path(agent.instructions_file).read_text(encoding='utf-8')

    raise AgentError('agent has neither instructions nor instructions_file')


AGENTS = SectionSpec(
    'agents',
    AgentConfigList,
    required=True,
    describe=_describe_agents,
)

DEFAULTS = SectionSpec('defaults', DefaultsConfig)
