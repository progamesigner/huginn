from typing import override

from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.providers.config import ConfigProvider
from app.core.providers.llm import LLMProvider
from app.core.providers.temporal import ClockProvider
from lib.container import ContainerContract, ProviderContract
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
    dependencies = (ClockProvider, ConfigProvider, LLMProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(ToolSourceRegistry, instance=ToolSourceRegistry())
        container.bind(Router, factory=_router)

        sections = container.resolve(SectionRegistry)
        for spec in build_sections():
            sections.register(spec)


class TurnProvider(ProviderContract):
    dependencies = (AgentProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(TurnHandlerContract, factory=_turns)
