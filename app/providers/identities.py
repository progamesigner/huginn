from typing import override

from app.core.config.registry import HuginnConfig
from app.features.auth.config import API_KEYS
from app.features.auth.providers import AuthProvider
from app.features.channels.config import CHANNELS
from app.features.channels.providers import ChannelProvider
from app.features.identities.contracts import IdentityTargetServiceContract
from app.features.identities.providers import IdentityProvider
from lib.container import ContainerContract, ProviderContract


class ConfiguredIdentityTargets(IdentityTargetServiceContract):
    def __init__(self, config: HuginnConfig) -> None:
        self.config = config

    @override
    def resolve(self, target: str) -> tuple[str, str]:
        channels = self.config.section(CHANNELS).root
        keys = self.config.section(API_KEYS).root
        # Prefixes disambiguate a channel and an application with the same name.
        if target.startswith('http:') and target[5:] in keys:
            return 'http', target[5:]
        if target.startswith('channel:') and target[8:] in channels:
            name = target[8:]
            return channels[name].kind, name
        if target in channels and target not in keys:
            return channels[target].kind, target
        if target in keys and target not in channels:
            return 'http', target
        raise ValueError(
            'unknown or ambiguous target; use channel:<name> or http:<key>',
        )


class IdentityTargetsProvider(ProviderContract):
    dependencies = (IdentityProvider, ChannelProvider, AuthProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            IdentityTargetServiceContract,
            factory=lambda c: ConfiguredIdentityTargets(c.resolve(HuginnConfig)),
            shared=True,
        )
