from collections.abc import AsyncIterator
from typing import Annotated, Literal, Protocol

from pydantic import Field, SecretStr
from structlog.stdlib import BoundLogger

from .types import (
    CompletionChunk,
    CompletionRequest,
    LLMValue,
    ModelCapabilities,
    ModelRef,
)

type Adapter = Literal['openai', 'anthropic', 'gemini']


class ProviderConfig(LLMValue):
    name: Annotated[str, Field()]
    adapter: Annotated[Adapter, Field()]
    api_key: Annotated[SecretStr, Field()]
    models: Annotated[tuple[str, ...], Field()]
    base_url: Annotated[str | None, Field()] = None
    # NOTE: opt-in per model, not per provider
    cache_control: Annotated[tuple[str, ...], Field()] = ()
    vision: Annotated[tuple[str, ...], Field()] = ()
    documents: Annotated[tuple[str, ...], Field()] = ()

    def caches(self, model: str) -> bool:
        return model in self.cache_control

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(
            vision=model in self.vision,
            documents=model in self.documents,
        )


class LLMClient(Protocol):
    def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]: ...


class ProviderRegistry:
    providers: dict[str, ProviderConfig]
    models: dict[str, tuple[ProviderConfig, ...]]

    def __init__(
        self,
        providers: tuple[ProviderConfig, ...],
        logger: BoundLogger,
    ) -> None:
        self.providers = {provider.name: provider for provider in providers}
        self.models = _index_models(providers, logger)

    def resolve(self, ref: ModelRef) -> ProviderConfig | None:
        if ref.provider is not None:
            provider = self.providers.get(ref.provider)
            if provider is None or ref.model not in provider.models:
                return None

            return provider

        match self.models.get(ref.model):
            case (provider,):
                return provider
            case _:
                return None


def _index_models(
    providers: tuple[ProviderConfig, ...],
    logger: BoundLogger,
) -> dict[str, tuple[ProviderConfig, ...]]:
    claims = {
        model: tuple(owner for owner in providers if model in owner.models)
        for provider in providers
        for model in provider.models
    }

    for model, owners in sorted(claims.items()):
        if len(owners) > 1:
            logger.warning(
                'Model %r is claimed by several providers (%s); '
                'it can only be reached by naming one of them',
                model,
                ', '.join(owner.name for owner in owners),
            )

    return claims
