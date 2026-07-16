import sys

import pytest
from pydantic import SecretStr

from lib.container import Disposable
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
    manager = LLMManager((anthropic,))

    assert manager.provider_for('claude-opus-4-8') is anthropic
    assert manager.provider_for('unknown-model') is None


def test_manager_satisfies_disposable_contract() -> None:
    manager = LLMManager(())

    assert isinstance(manager, Disposable)


async def test_manager_dispose_does_not_raise() -> None:
    manager = LLMManager(())

    await manager.dispose()


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
    manager = LLMManager((anthropic, openai))

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
    manager = LLMManager((provider,))

    assert manager.client_for('gpt-5') is manager.client_for('gpt-5')


async def test_dispose_closes_cached_adapter_clients() -> None:
    provider = ProviderConfig(
        name='openai',
        adapter='openai',
        api_key=SecretStr('secret'),
        models=('gpt-5',),
    )
    manager = LLMManager((provider,))
    manager.client_for('gpt-5')

    await manager.dispose()


def test_client_for_reports_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, 'lib.llm.adapters.openai', None)
    provider = ProviderConfig(
        name='openai',
        adapter='openai',
        api_key=SecretStr('secret'),
        models=('gpt-5',),
    )
    manager = LLMManager((provider,))

    with pytest.raises(ModuleNotFoundError, match=r'huginn\[openai\]'):
        manager.client_for('gpt-5')
