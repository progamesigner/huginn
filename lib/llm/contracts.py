from abc import ABC, abstractmethod

from .client import LLMClient, ProviderConfig


class LLMContract(ABC):
    @abstractmethod
    def client_for(self, model: str) -> LLMClient | None: ...

    @abstractmethod
    def provider_for(self, model: str) -> ProviderConfig | None: ...
