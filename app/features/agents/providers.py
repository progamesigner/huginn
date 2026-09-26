from datetime import timedelta
from typing import override

from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.providers.command import CommandProvider
from app.core.providers.config import ConfigProvider
from app.core.providers.decision import DecisionProvider
from app.core.providers.llm import LLMProvider
from app.core.providers.temporal import ClockProvider
from lib.command import CommandContract
from lib.concurrency import TaskSupervisor
from lib.container import ContainerContract, ProviderContract
from lib.decision import DecisionContract
from lib.llm import LLMContract
from lib.temporal import ClockContract

from .config import AGENTS, DEFAULTS, DefaultsConfig, build_sections
from .contracts import (
    ToolSourceRegistry,
    TranscriptStoreContract,
    TurnHandlerContract,
)
from .router import Router
from .services import TurnHandler

# NOTE: a shadow decision is advisory, so only a few run at once against a slow
# gateway and shutdown does not wait long for the rest
_SHADOW_LIMIT = 8
_SHADOW_GRACE = timedelta(seconds=5)


def _defaults(container: ContainerContract) -> DefaultsConfig:
    return container.resolve(HuginnConfig).section(DEFAULTS)


def _router(container: ContainerContract) -> Router:
    config = container.resolve(HuginnConfig)
    defaults = _defaults(container)
    return Router(
        config.section(AGENTS).root,
        container.resolve(LLMContract),
        defaults,
        defaults.router_prompt,
        decisions=container.resolve(DecisionContract),
        shadows=TaskSupervisor(limit=_SHADOW_LIMIT, grace=_SHADOW_GRACE),
    )


def _turns(container: ContainerContract) -> TurnHandler:
    defaults = _defaults(container)
    return TurnHandler(
        llm=container.resolve(LLMContract),
        router=container.resolve(Router),
        sources=container.resolve(ToolSourceRegistry),
        store=container.resolve(TranscriptStoreContract),
        clock=container.resolve(ClockContract),
        defaults=defaults,
        default_max_completion_tokens=defaults.max_completion_tokens,
    )


class AgentProvider(ProviderContract):
    dependencies = (ClockProvider, ConfigProvider, LLMProvider, DecisionProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ToolSourceRegistry, instance=ToolSourceRegistry())
        container.bind(Router, factory=_router, shared=True)

        sections = container.resolve(SectionRegistry)
        for spec in build_sections():
            sections.register(spec)


class TurnProvider(ProviderContract):
    dependencies = (AgentProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(TurnHandlerContract, factory=_turns)


class RouterCommandProvider(ProviderContract):
    dependencies = (AgentProvider, CommandProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from .router_commands import cli

        container.resolve(CommandContract).mount('router', cli)
