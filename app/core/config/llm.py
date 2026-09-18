from typing import Annotated

from pydantic import Field, RootModel, SecretStr

from app.core.config.exceptions import (
    UnknownCacheControlModelError,
    UnknownCapabilityModelError,
)
from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue
from lib.llm import Adapter


class ProviderConfig(ConfigValue):
    adapter: Annotated[Adapter, Field()]
    api_key: Annotated[SecretStr, Field()]
    models: Annotated[tuple[str, ...], Field()]
    base_url: Annotated[str | None, Field()] = None
    cache_control: Annotated[tuple[str, ...], Field()] = ()
    vision: Annotated[tuple[str, ...], Field()] = ()
    documents: Annotated[tuple[str, ...], Field()] = ()


class ProviderConfigList(RootModel[dict[str, ProviderConfig]]):
    pass


def _describe_providers(config: HuginnConfig) -> list[str]:
    providers = config.section(PROVIDERS).root
    entries = ', '.join(
        f'{name} ({", ".join(provider.models)})'
        for name, provider in sorted(providers.items())
    )
    cached = ', '.join(
        f'{name} ({", ".join(provider.cache_control)})'
        for name, provider in sorted(providers.items())
        if provider.cache_control
    )
    return [
        f'providers: {entries or "none"}',
        f'cache_control: {cached or "none"}',
    ]


def _validate_providers(config: HuginnConfig) -> None:
    for name, provider in config.section(PROVIDERS).root.items():
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
