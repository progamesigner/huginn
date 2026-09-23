import logging
from typing import Annotated, Literal, Protocol, Self

from pydantic import Field, SecretStr, model_validator

from .types import DecisionRef, DecisionRequest, DecisionResult, DecisionValue

logger = logging.getLogger(__name__)

type Adapter = Literal['bifrost', 'typesafe']

_DEFAULT_TIMEOUT = 5.0

# NOTE: TypeSafe's own format is the default wire; anything that speaks it, such as
# OpenRouter, is reached by url alone. A Bifrost gateway is self-hosted, so it has no
# address to fall back to
DEFAULT_URLS: dict[Adapter, str] = {
    'typesafe': 'https://api.typesafe.ai/v1/systemone',
}


class DecisionProviderConfig(DecisionValue):
    name: Annotated[str, Field()]
    adapter: Annotated[Adapter, Field()]
    api_key: Annotated[SecretStr, Field()]
    models: Annotated[tuple[str, ...], Field(min_length=1)]
    url: Annotated[str | None, Field()] = None
    timeout: Annotated[float, Field(gt=0)] = _DEFAULT_TIMEOUT

    @model_validator(mode='after')
    def _addressable(self) -> Self:
        if self.url is None and self.adapter not in DEFAULT_URLS:
            raise ValueError(f'the {self.adapter!r} adapter needs a url')

        return self

    @property
    def endpoint(self) -> str:
        return self.url or DEFAULT_URLS[self.adapter]


class DecisionClient(Protocol):
    async def decide(self, request: DecisionRequest) -> DecisionResult: ...


class DecisionProviderRegistry:
    providers: dict[str, DecisionProviderConfig]
    models: dict[str, tuple[DecisionProviderConfig, ...]]

    def __init__(self, providers: tuple[DecisionProviderConfig, ...]) -> None:
        self.providers = {provider.name: provider for provider in providers}
        self.models = _index_models(providers)

    def resolve(self, ref: DecisionRef) -> DecisionProviderConfig | None:
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
    providers: tuple[DecisionProviderConfig, ...],
) -> dict[str, tuple[DecisionProviderConfig, ...]]:
    claims = {
        model: tuple(owner for owner in providers if model in owner.models)
        for provider in providers
        for model in provider.models
    }

    for model, owners in sorted(claims.items()):
        if len(owners) > 1:
            logger.warning(
                'Decision model %r is claimed by several providers (%s); '
                'it can only be reached by naming one of them',
                model,
                ', '.join(owner.name for owner in owners),
            )

    return claims
