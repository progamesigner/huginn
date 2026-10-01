from typing import override

from app.core.config.llm import ProviderConfig
from app.core.config.providers import llm_providers
from app.core.config.registry import HuginnConfig
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract
from lib.llm import LLMContract, LLMManager
from lib.llm import ProviderConfig as LLMProviderConfig

from .config import ConfigProvider
from .http import HTTPClientProvider
from .settings import SettingsProvider


def _build_providers(
    providers: dict[str, ProviderConfig],
) -> tuple[LLMProviderConfig, ...]:
    return tuple(
        LLMProviderConfig(
            name=name,
            adapter=provider.adapter,
            api_key=provider.api_key,
            models=provider.models,
            base_url=provider.base_url,
            cache_control=provider.cache_control,
            vision=provider.vision,
            documents=provider.documents,
        )
        for name, provider in providers.items()
    )


class LLMProvider(ProviderContract):
    dependencies = (SettingsProvider, HTTPClientProvider, ConfigProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            LLMContract,
            factory=self._build,
            shared=True,
        )

    def _build(self, container: ContainerContract) -> LLMManager:
        config = container.resolve(HuginnConfig)
        return LLMManager(
            _build_providers(llm_providers(config)),
            container.resolve(HTTPClientContract),
        )
