from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from httpx2 import URL, AsyncClient

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


# NOTE: structural so any HTTP client manager fits without lib/llm importing it
class HTTPConnector(Protocol):
    def connect(
        self,
        base_url: URL | str = '',
        headers: Mapping[str, str] | None = None,
    ) -> AbstractAsyncContextManager[AsyncClient]: ...
