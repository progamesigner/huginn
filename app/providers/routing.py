from typing import override

from app.core.config.registry import HuginnConfig
from app.core.providers.command import CommandProvider
from app.features.agents.entities import (
    RoutingPolicy,
    RoutingTarget,
    RoutingTargets,
    SelectionPolicy,
)
from app.features.agents.providers import RouterCommandProvider
from app.features.auth.config import API_KEYS
from app.features.channels.config import CHANNELS, routing_policy
from lib.container import ContainerContract, ProviderContract


def _targets(container: ContainerContract) -> RoutingTargets:
    config = container.resolve(HuginnConfig)
    entries = {
        f'channels.{name}': RoutingTarget(channel=name, policy=routing_policy(channel))
        for name, channel in config.section(CHANNELS).root.items()
    }
    for name, key in config.section(API_KEYS).root.items():
        selection = (
            None
            if key.router is None
            else SelectionPolicy(
                fallback=key.router.fallback,
                engine=key.router.engine,
                model=key.router.model,
                provider=key.router.provider,
                allowed=frozenset(key.router.allowed),
                history_turns=key.router.history_turns,
            )
        )
        entries[f'api_keys.{name}'] = RoutingTarget(
            channel=name,
            policy=RoutingPolicy(
                agent=key.agent,
                selection=selection,
                allowed=frozenset(key.agents),
            ),
        )
    return RoutingTargets(entries=entries)


class RoutingCommandProvider(ProviderContract):
    dependencies = (CommandProvider, RouterCommandProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(RoutingTargets, factory=_targets, shared=True)
