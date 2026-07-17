from collections.abc import AsyncIterator
from typing import Annotated, Literal, Protocol

from pydantic import Field, SecretStr

from .types import CompletionChunk, CompletionRequest, LLMValue

type Adapter = Literal['openai', 'anthropic']


class ProviderConfig(LLMValue):
    name: Annotated[str, Field()]
    adapter: Annotated[Adapter, Field()]
    api_key: Annotated[SecretStr, Field()]
    models: Annotated[tuple[str, ...], Field()]
    base_url: Annotated[str | None, Field()] = None


class LLMClient(Protocol):
    def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]: ...

    async def dispose(self) -> None: ...


class ProviderRegistry:
    models: dict[str, ProviderConfig]

    def __init__(self, providers: tuple[ProviderConfig, ...]) -> None:
        self.models = {
            model: provider for provider in providers for model in provider.models
        }

    def resolve(self, model: str) -> ProviderConfig | None:
        return self.models.get(model)
