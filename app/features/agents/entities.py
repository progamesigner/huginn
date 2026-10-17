import re
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated

from pydantic import Field, JsonValue

from app.contracts.auth import RouterEngine
from app.contracts.channels import AttachmentKind, TurnEnding
from app.core.schema import AppEntity, AppValue
from lib.llm import Message, ModelCapabilities

from .config import AgentConfig, BindingSource, DefaultsConfig, RouterConfig

_DEFAULT_MAX_ITERATIONS = 10
_DEFAULT_TURN_TIMEOUT = 300.0
_DEFAULT_TOOL_TIMEOUT = 60.0
_DEFAULT_MIN_ITERATION_BUDGET = 15.0


class ShadowPolicy(AppValue):
    model: Annotated[str, Field()]
    provider: Annotated[str | None, Field()] = None


class SelectionPolicy(AppValue):
    fallback: Annotated[str, Field()]
    engine: Annotated[RouterEngine, Field()] = 'llm'
    history_turns: Annotated[int | None, Field()] = None
    model: Annotated[str | None, Field()] = None
    provider: Annotated[str | None, Field()] = None
    allowed: Annotated[frozenset[str], Field()] = frozenset()
    min_confidence: Annotated[float | None, Field()] = None
    min_margin: Annotated[float | None, Field()] = None
    shadow: Annotated[ShadowPolicy | None, Field()] = None

    @property
    def history_limit(self) -> int:
        if self.history_turns is not None:
            return self.history_turns
        return 6 if self.engine == 'decision' else 0


def selection_policy(router: RouterConfig) -> SelectionPolicy:
    return SelectionPolicy(
        fallback=router.fallback,
        engine=router.engine,
        history_turns=router.history_turns,
        model=router.model,
        provider=router.provider,
        allowed=frozenset(router.allowed),
        min_confidence=router.min_confidence,
        min_margin=router.min_margin,
        shadow=(
            None
            if router.shadow is None
            else ShadowPolicy(
                model=router.shadow.model,
                provider=router.shadow.provider,
            )
        ),
    )


class RoutingPolicy(AppValue):
    agent: Annotated[str | None, Field()] = None
    selection: Annotated[SelectionPolicy | None, Field()] = None
    allowed: Annotated[frozenset[str], Field()] = frozenset()


class RoutedAgent(AppValue):
    name: Annotated[str, Field()]
    config: Annotated[AgentConfig, Field()]


class PromptAttachment(AppValue):
    kind: Annotated[AttachmentKind, Field()]
    mime: Annotated[str, Field()]
    reference: Annotated[str, Field()]
    unique: Annotated[str, Field()]
    data: Annotated[bytes, Field()]
    hint: Annotated[str, Field()]
    name: Annotated[str | None, Field()] = None
    transcript: Annotated[str | None, Field()] = None
    document_text: Annotated[str | None, Field()] = None


class TurnPrompt(AppValue):
    text: Annotated[str, Field()]
    actor: Annotated[str, Field()]
    attachments: Annotated[tuple[PromptAttachment, ...], Field()] = ()
    account: Annotated[str | None, Field()] = None
    handle: Annotated[str | None, Field()] = None
    memory_scope: Annotated[str | None, Field()] = None


class RoutingTurn(AppValue):
    agent: Annotated[str, Field()]
    user: Annotated[tuple[str, ...], Field()]
    answer: Annotated[str, Field()]
    attachments: Annotated[tuple[str, ...], Field()] = ()


class RoutingContext(AppValue):
    current: Annotated[tuple[TurnPrompt, ...], Field()]
    turns: Annotated[tuple[RoutingTurn, ...], Field()] = ()
    summary: Annotated[str | None, Field()] = None


class TurnRequest(AppValue):
    channel: Annotated[str, Field()]
    chat: Annotated[str, Field()]
    policy: Annotated[RoutingPolicy, Field()]
    prompts: Annotated[tuple[TurnPrompt, ...], Field()]
    denied_tools: Annotated[frozenset[str], Field()] = frozenset()
    max_completion_tokens: Annotated[int | None, Field()] = None


class TurnScope(AppValue):
    actor: Annotated[str, Field()]
    agent: Annotated[str, Field()]
    channel: Annotated[str, Field()]
    chat: Annotated[str, Field()]
    account: Annotated[str | None, Field()] = None
    handle: Annotated[str | None, Field()] = None
    memory_scope: Annotated[str | None, Field()] = None
    document_blobs: Annotated[frozenset[str], Field()] = frozenset()


_SAFE_BOUND_VALUE = re.compile(r'^[A-Za-z0-9._:-]+$')


def resolve_bindings(
    inject: Mapping[str, JsonValue],
    bind: Mapping[str, BindingSource],
    scope: TurnScope,
) -> dict[str, JsonValue] | None:
    resolved = dict(inject)
    for key, source in bind.items():
        value = getattr(scope, source)
        # Keep deployed `user: handle` configurations on the account's approved
        # memory binding while preserving handle for non-memory arguments.
        if key == 'user' and source == 'handle':
            value = scope.memory_scope or scope.handle
        if not value or not _SAFE_BOUND_VALUE.fullmatch(value):
            return None
        resolved[key] = value
    return resolved


class ToolExecution(AppValue):
    name: Annotated[str, Field()]
    duration: Annotated[float, Field()]
    ok: Annotated[bool, Field()]


class Step(AppValue):
    tools: Annotated[tuple[ToolExecution, ...], Field()] = ()


class TurnBudget(AppValue):
    max_iterations: Annotated[int, Field()] = _DEFAULT_MAX_ITERATIONS
    turn_timeout: Annotated[float, Field()] = _DEFAULT_TURN_TIMEOUT
    tool_timeout: Annotated[float, Field()] = _DEFAULT_TOOL_TIMEOUT
    min_iteration_budget: Annotated[float, Field()] = _DEFAULT_MIN_ITERATION_BUDGET


def resolve_budget(agent: AgentConfig, defaults: DefaultsConfig) -> TurnBudget:
    default = TurnBudget()
    return TurnBudget(
        max_iterations=(
            agent.max_iterations or defaults.max_iterations or default.max_iterations
        ),
        turn_timeout=(
            agent.turn_timeout or defaults.turn_timeout or default.turn_timeout
        ),
        tool_timeout=agent.tool_timeout or default.tool_timeout,
    )


class TurnInput(AppValue):
    instructions: Annotated[str, Field()]
    prompts: Annotated[tuple[TurnPrompt, ...], Field()]
    model: Annotated[str, Field()]
    now: Annotated[datetime, Field()]
    channel: Annotated[str, Field()]
    capabilities: Annotated[ModelCapabilities, Field()]
    prior_turns: Annotated[tuple[Message, ...], Field()] = ()
    max_completion_tokens: Annotated[int | None, Field()] = None
    budget: Annotated[TurnBudget, Field()] = Field(default_factory=TurnBudget)


class TurnOutput(AppValue):
    text: Annotated[str, Field()]
    ending: Annotated[TurnEnding, Field()] = 'complete'
    steps: Annotated[tuple[Step, ...], Field()] = ()

    @property
    def completed(self) -> bool:
        return self.ending == 'complete'


class RoutingHistory(AppEntity):
    id: Annotated[str, Field()]
    historical_agent: Annotated[str, Field()]
    context: Annotated[RoutingContext, Field()]
    created_at: Annotated[datetime | None, Field()] = None
    previous_at: Annotated[datetime | None, Field()] = None


class RoutingTarget(AppValue):
    channel: Annotated[str, Field()]
    policy: Annotated[RoutingPolicy, Field()]


class RoutingTargets(AppValue):
    entries: Annotated[Mapping[str, RoutingTarget], Field()]
