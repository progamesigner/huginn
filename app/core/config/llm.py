from typing import Annotated

from pydantic import Field, RootModel, SecretStr

from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue
from lib.llm import Adapter


class ProviderConfig(ConfigValue):
    adapter: Annotated[Adapter, Field()]
    api_key: Annotated[SecretStr, Field()]
    models: Annotated[tuple[str, ...], Field()]
    base_url: Annotated[str | None, Field()] = None


class ProviderConfigList(RootModel[dict[str, ProviderConfig]]):
    pass


def _describe_providers(config: HuginnConfig) -> list[str]:
    providers = config.section(PROVIDERS).root
    entries = ', '.join(
        f'{name} ({", ".join(provider.models)})'
        for name, provider in sorted(providers.items())
    )
    return [f'providers: {entries or "none"}']


PROVIDERS = SectionSpec(
    'providers',
    ProviderConfigList,
    describe=_describe_providers,
)
