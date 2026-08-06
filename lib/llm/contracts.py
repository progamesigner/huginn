from abc import ABC, abstractmethod

from .client import LLMClient, ProviderConfig
from .types import ModelRef


class LLMContract(ABC):
    @abstractmethod
    def client_for(self, ref: ModelRef) -> LLMClient | None: ...

    @abstractmethod
    def provider_for(self, ref: ModelRef) -> ProviderConfig | None: ...
