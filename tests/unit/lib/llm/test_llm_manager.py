import sys

import pytest
from pydantic import SecretStr

from lib.http import HTTPClientManager
from lib.llm import LLMManager, ProviderConfig
from lib.llm.adapters.anthropic import AnthropicAdapter
from lib.llm.adapters.openai import OpenAIAdapter


def test_manager_looks_up_provider_by_model() -> None:
    anthropic = ProviderConfig(
        name='anthropic',
        adapter='anthropic',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8',),
    )
    manager = LLMManager((anthropic,), HTTPClientManager())

    assert manager.provider_for('claude-opus-4-8') is anthropic
    assert manager.provider_for('unknown-model') is None


def test_client_for_builds_matching_adapter_per_provider_tag() -> None:
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
    manager = LLMManager((anthropic, openai), HTTPClientManager())

    assert isinstance(manager.client_for('claude-opus-4-8'), AnthropicAdapter)
    assert isinstance(manager.client_for('gpt-5'), OpenAIAdapter)
    assert manager.client_for('unknown-model') is None


def test_client_for_caches_adapter_per_provider() -> None:
    provider = ProviderConfig(
        name='openai',
        adapter='openai',
        api_key=SecretStr('secret'),
        models=('gpt-5',),
    )
    manager = LLMManager((provider,), HTTPClientManager())

    assert manager.client_for('gpt-5') is manager.client_for('gpt-5')


@pytest.mark.parametrize(
    ('provider', 'model'),
    [
        (
            ProviderConfig(
                name='openai',
                adapter='openai',
                api_key=SecretStr('secret'),
                models=('gpt-5',),
            ),
            'gpt-5',
        ),
        (
            ProviderConfig(
                name='anthropic',
                adapter='anthropic',
                api_key=SecretStr('secret'),
                models=('claude-opus-4-8',),
            ),
            'claude-opus-4-8',
        ),
    ],
)
def test_client_for_reports_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
    provider: ProviderConfig,
    model: str,
) -> None:
    monkeypatch.setitem(sys.modules, f'lib.llm.adapters.{provider.adapter}', None)
    manager = LLMManager((provider,), HTTPClientManager())

    with pytest.raises(ModuleNotFoundError, match=rf'huginn\[{provider.adapter}\]'):
        manager.client_for(model)
