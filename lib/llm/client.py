from collections.abc import AsyncIterator
from typing import Annotated, Literal, Protocol

from pydantic import Field, SecretStr

from .types import BaseValue, CompletionChunk, CompletionRequest

type Adapter = Literal['openai', 'anthropic']


class ProviderConfig(BaseValue):
    name: Annotated[str, Field()]
    adapter: Annotated[Adapter, Field()]
    api_key: Annotated[SecretStr, Field()]
    models: Annotated[tuple[str, ...], Field()]
    base_url: Annotated[str | None, Field()] = None


class LLMClient(Protocol):
    def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionChunk]: ...


class ProviderRegistry:
    models: dict[str, ProviderConfig]

    def __init__(self, providers: tuple[ProviderConfig, ...]) -> None:
        self.models = {
            model: provider for provider in providers for model in provider.models
        }

    def resolve(self, model: str) -> ProviderConfig | None:
        return self.models.get(model)
