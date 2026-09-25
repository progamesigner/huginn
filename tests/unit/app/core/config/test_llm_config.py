import pytest
from pydantic import SecretStr, ValidationError

from app.core.config.exceptions import (
    UnknownCacheControlModelError,
    UnknownCapabilityModelError,
)
from app.core.config.llm import ProviderConfig
from app.core.config.providers import PROVIDERS, ProviderConfigList
from app.core.config.registry import HuginnConfig


def test_provider_config_holds_named_declarations() -> None:
    providers = ProviderConfigList(
        {
            'openai': ProviderConfig(
                adapter='openai',
                api_key=SecretStr('sk-test'),
                models=('gpt-5-mini',),
            ),
        },
    )

    assert providers.root['openai'].adapter == 'openai'
    assert providers.root['openai'].models == ('gpt-5-mini',)


def test_provider_config_requires_adapter_api_key_and_models() -> None:
    with pytest.raises(ValidationError, match='adapter'):
        ProviderConfig.model_validate({'api_key': 'sk-test', 'models': ['gpt-5-mini']})

    with pytest.raises(ValidationError, match='api_key'):
        ProviderConfig.model_validate({'adapter': 'openai', 'models': ['gpt-5-mini']})

    with pytest.raises(ValidationError, match='models'):
        ProviderConfig.model_validate({'adapter': 'openai', 'api_key': 'sk-test'})


def test_provider_config_rejects_unknown_adapter() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(
            {'adapter': 'mystery', 'api_key': 'sk-test', 'models': ['gpt-5-mini']},
        )


def test_provider_config_defaults_cache_control_to_no_models() -> None:
    provider = ProviderConfig(
        adapter='openai',
        api_key=SecretStr('sk-test'),
        models=('gpt-5-mini',),
    )

    assert provider.cache_control == ()
    assert provider.vision == ()
    assert provider.documents == ()


def _config(cache_control: tuple[str, ...]) -> HuginnConfig:
    return HuginnConfig(
        sections={
            PROVIDERS.key: ProviderConfigList(
                {
                    'bifrost': ProviderConfig(
                        adapter='openai',
                        api_key=SecretStr('sk-test'),
                        models=('anthropic/claude-sonnet-5', 'openai/gpt-5.6-sol'),
                        cache_control=cache_control,
                    ),
                },
            ),
        },
    )


def test_validate_providers_accepts_cache_control_drawn_from_models() -> None:
    assert PROVIDERS.validate is not None

    PROVIDERS.validate(_config(('anthropic/claude-sonnet-5',)))


def test_validate_providers_rejects_a_cache_control_model_it_does_not_serve() -> None:
    assert PROVIDERS.validate is not None

    with pytest.raises(UnknownCacheControlModelError, match='anthropic/claude-opus-5'):
        PROVIDERS.validate(_config(('anthropic/claude-opus-5',)))


def test_validate_providers_rejects_an_unknown_capability_model() -> None:
    provider = ProviderConfig(
        adapter='openai',
        api_key=SecretStr('sk-test'),
        models=('gpt-5-mini',),
        vision=('gpt-5-vision',),
    )
    config = HuginnConfig(
        sections={PROVIDERS.key: ProviderConfigList({'openai': provider})},
    )

    assert PROVIDERS.validate is not None
    with pytest.raises(UnknownCapabilityModelError, match='gpt-5-vision'):
        PROVIDERS.validate(config)


def test_provider_config_rejects_name_field() -> None:
    # NOTE: the provider's name comes from the providers.<name> mapping key, not a field
    with pytest.raises(ValidationError, match='name'):
        ProviderConfig.model_validate(
            {
                'name': 'openai',
                'adapter': 'openai',
                'api_key': 'sk-test',
                'models': ['gpt-5-mini'],
            },
        )
