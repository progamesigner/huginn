from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from httpx2 import URL, AsyncClient

from .client import DecisionClient, DecisionProviderConfig
from .types import DecisionRef


class DecisionContract(ABC):
    @abstractmethod
    def client_for(self, ref: DecisionRef) -> DecisionClient | None: ...

    @abstractmethod
    def provider_for(self, ref: DecisionRef) -> DecisionProviderConfig | None: ...


# NOTE: structural so any HTTP client manager fits without lib/decision importing it
class HTTPConnector(Protocol):
    def connect(
        self,
        base_url: URL | str = '',
        headers: Mapping[str, str] | None = None,
    ) -> AbstractAsyncContextManager[AsyncClient]: ...
