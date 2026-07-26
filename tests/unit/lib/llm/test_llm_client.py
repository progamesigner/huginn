import pytest
from pydantic import SecretStr, ValidationError

from lib.llm import ProviderConfig, ProviderRegistry


def test_provider_config_defaults_to_direct_connection() -> None:
    provider = ProviderConfig(
        name='anthropic',
        adapter='anthropic',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8',),
    )

    assert provider.base_url is None


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

    resolved = registry.resolve('claude-opus-4-8')

    assert resolved is not None
    assert resolved.adapter == 'anthropic'
    assert registry.resolve('gpt-5') is openai


def test_registry_returns_none_for_unknown_model() -> None:
    registry = ProviderRegistry(())

    assert registry.resolve('unknown-model') is None
