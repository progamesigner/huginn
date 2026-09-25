from typing import override

from app.core.config.decision import DecisionProviderConfig as DecisionProviderSettings
from app.core.config.providers import decision_providers
from app.core.config.registry import HuginnConfig
from lib.container import ContainerContract, ProviderContract
from lib.decision import DecisionContract, DecisionManager, DecisionProviderConfig
from lib.http import HTTPClientContract

from .config import ProvidersConfigProvider
from .http import HTTPClientProvider
from .settings import SettingsProvider


def _build_providers(
    providers: dict[str, DecisionProviderSettings],
) -> tuple[DecisionProviderConfig, ...]:
    return tuple(
        DecisionProviderConfig(
            name=name,
            adapter=provider.adapter,
            api_key=provider.api_key,
            models=provider.models,
            url=provider.url,
            timeout=provider.timeout,
        )
        for name, provider in providers.items()
    )


class DecisionProvider(ProviderContract):
    dependencies = (SettingsProvider, HTTPClientProvider, ProvidersConfigProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            DecisionContract,
            factory=self._build,
            shared=True,
        )

    def _build(self, container: ContainerContract) -> DecisionManager:
        config = container.resolve(HuginnConfig)
        return DecisionManager(
            _build_providers(decision_providers(config)),
            container.resolve(HTTPClientContract),
        )
