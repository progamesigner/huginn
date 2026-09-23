from abc import ABC, abstractmethod

from .client import DecisionClient, DecisionProviderConfig
from .types import DecisionRef


class DecisionContract(ABC):
    @abstractmethod
    def client_for(self, ref: DecisionRef) -> DecisionClient | None: ...

    @abstractmethod
    def provider_for(self, ref: DecisionRef) -> DecisionProviderConfig | None: ...
