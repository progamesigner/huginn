from typing import Annotated

from pydantic import Discriminator, RootModel, Tag

from app.core.config.decision import DecisionProviderConfig
from app.core.config.exceptions import (
    UnknownCacheControlModelError,
    UnknownCapabilityModelError,
)
from app.core.config.llm import ProviderConfig
from app.core.config.registry import HuginnConfig, SectionSpec


def _type(value: object) -> str:
    # NOTE: a provider without a type is an llm one, so configs written before
    # decision providers existed read unchanged
    found = (
        value.get('type') if isinstance(value, dict) else getattr(value, 'type', None)
    )
    return 'llm' if found is None else str(found)


type AnyProviderConfig = Annotated[
    Annotated[ProviderConfig, Tag('llm')]
    | Annotated[DecisionProviderConfig, Tag('decision')],
    Discriminator(_type),
]


class ProviderConfigList(RootModel[dict[str, AnyProviderConfig]]):
    pass


def llm_providers(config: HuginnConfig) -> dict[str, ProviderConfig]:
    return {
        name: provider
        for name, provider in config.section(PROVIDERS).root.items()
        if isinstance(provider, ProviderConfig)
    }


def decision_providers(config: HuginnConfig) -> dict[str, DecisionProviderConfig]:
    return {
        name: provider
        for name, provider in config.section(PROVIDERS).root.items()
        if isinstance(provider, DecisionProviderConfig)
    }


def _describe_providers(config: HuginnConfig) -> list[str]:
    llm = llm_providers(config)
    entries = ', '.join(
        f'{name} ({", ".join(provider.models)})'
        for name, provider in sorted(llm.items())
    )
    decisions = ', '.join(
        f'{name} ({", ".join(provider.models)})'
        for name, provider in sorted(decision_providers(config).items())
    )
    cached = ', '.join(
        f'{name} ({", ".join(provider.cache_control)})'
        for name, provider in sorted(llm.items())
        if provider.cache_control
    )
    return [
        f'providers: {entries or "none"}',
        f'decision providers: {decisions or "none"}',
        f'cache_control: {cached or "none"}',
    ]


def _validate_providers(config: HuginnConfig) -> None:
    for name, provider in llm_providers(config).items():
        unknown = sorted(set(provider.cache_control) - set(provider.models))
        if unknown:
            raise UnknownCacheControlModelError(
                f'providers.{name}.cache_control: '
                f'{", ".join(unknown)} not listed in models',
            )
        capabilities = {'vision': provider.vision, 'documents': provider.documents}
        for capability, models in capabilities.items():
            if unknown := sorted(set(models) - set(provider.models)):
                raise UnknownCapabilityModelError(
                    f'providers.{name}.{capability}: '
                    f'{", ".join(unknown)} not listed in models',
                )


PROVIDERS = SectionSpec(
    'providers',
    ProviderConfigList,
    describe=_describe_providers,
    validate=_validate_providers,
)
