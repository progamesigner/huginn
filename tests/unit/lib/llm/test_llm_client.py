import pytest
from pydantic import SecretStr, ValidationError

from lib.llm import ModelRef, ProviderConfig, ProviderRegistry


def test_provider_config_defaults_to_direct_connection() -> None:
    provider = ProviderConfig(
        name='anthropic',
        adapter='anthropic',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8',),
    )

    assert provider.base_url is None


def test_provider_config_caches_only_the_models_that_opted_in() -> None:
    provider = ProviderConfig(
        name='bifrost',
        adapter='openai',
        api_key=SecretStr('secret'),
        models=('anthropic/claude-opus-5', 'openai/gpt-5.6-sol'),
        cache_control=('anthropic/claude-opus-5',),
    )

    assert provider.caches('anthropic/claude-opus-5')
    assert not provider.caches('openai/gpt-5.6-sol')


def test_provider_config_caches_nothing_by_default() -> None:
    provider = ProviderConfig(
        name='anthropic',
        adapter='anthropic',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8',),
    )

    assert not provider.caches('claude-opus-4-8')


def test_provider_config_reports_declared_model_capabilities() -> None:
    provider = ProviderConfig(
        name='anthropic',
        adapter='anthropic',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8',),
        vision=('claude-opus-4-8',),
    )

    assert provider.capabilities('claude-opus-4-8').vision
    assert not provider.capabilities('claude-opus-4-8').documents


def test_provider_config_requires_api_key() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(
            {
                'name': 'anthropic',
                'adapter': 'anthropic',
                'models': ('claude-opus-4-8',),
            },
        )


def test_registry_routes_model_to_its_provider_adapter() -> None:
    anthropic = ProviderConfig(
        name='anthropic',
        adapter='anthropic',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8',),
    )
    openai = ProviderConfig(
        name='openai',
        adapter='openai',
        api_key=SecretStr('secret'),
        models=('gpt-5',),
    )
    registry = ProviderRegistry((anthropic, openai))

    resolved = registry.resolve(ModelRef(model='claude-opus-4-8'))

    assert resolved is not None
    assert resolved.adapter == 'anthropic'
    assert registry.resolve(ModelRef(model='gpt-5')) is openai


def test_registry_returns_none_for_unknown_model() -> None:
    registry = ProviderRegistry(())

    assert registry.resolve(ModelRef(model='unknown-model')) is None
