from abc import ABC, abstractmethod

from .client import LLMClient, ProviderConfig
from .types import ModelCapabilities, ModelRef


class LLMContract(ABC):
    @abstractmethod
    def client_for(self, ref: ModelRef) -> LLMClient | None: ...

    @abstractmethod
    def provider_for(self, ref: ModelRef) -> ProviderConfig | None: ...

    def capabilities(self, ref: ModelRef) -> ModelCapabilities:
        provider = self.provider_for(ref)
        return (
            ModelCapabilities()
            if provider is None
            else provider.capabilities(ref.model)
        )
