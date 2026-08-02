import pytest
from pydantic import SecretStr, ValidationError

from app.core.config.llm import ProviderConfig, ProviderConfigList


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
