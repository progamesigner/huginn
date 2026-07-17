from typing import override

from app.core.settings import HuginnSettings
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientManager
from lib.llm import LLMManager

from .http import HTTPClientProvider
from .settings import SettingsProvider


class LLMProvider(ProviderContract):
    dependencies = (SettingsProvider, HTTPClientProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            LLMManager,
            factory=lambda c: LLMManager(
                c.resolve(HuginnSettings).providers,
                c.resolve(HTTPClientManager),
            ),
            shared=True,
        )
