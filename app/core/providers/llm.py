from typing import override

from app.core.settings import HuginnSettings
from lib.container import ContainerContract, ProviderContract
from lib.llm import LLMManager

from .settings import SettingsProvider


class LLMProvider(ProviderContract):
    dependencies = (SettingsProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            LLMManager,
            factory=lambda c: LLMManager(c.resolve(HuginnSettings).providers),
            shared=True,
        )
