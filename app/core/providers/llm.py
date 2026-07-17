from typing import override

from app.core.settings import HuginnSettings
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract
from lib.llm import LLMContract, LLMManager

from .http import HTTPClientProvider
from .settings import SettingsProvider


class LLMProvider(ProviderContract):
    dependencies = (SettingsProvider, HTTPClientProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            LLMContract,
            factory=lambda c: LLMManager(
                c.resolve(HuginnSettings).providers,
                c.resolve(HTTPClientContract),
            ),
            shared=True,
        )
