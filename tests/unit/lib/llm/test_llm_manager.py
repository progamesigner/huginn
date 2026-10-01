import sys

import pytest
from pydantic import SecretStr

from lib.http import HTTPClientManager
from lib.llm import LLMManager, ModelRef, ProviderConfig
from lib.llm.adapters.anthropic import AnthropicAdapter
from lib.llm.adapters.openai import OpenAIAdapter
from tests.helpers.logging import make_logger


def test_manager_looks_up_provider_by_model() -> None:
    anthropic = ProviderConfig(
        name='anthropic',
        adapter='anthropic',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8',),
    )
    manager = LLMManager((anthropic,), HTTPClientManager(), logger=make_logger())

    assert manager.provider_for(ModelRef(model='claude-opus-4-8')) is anthropic
    assert manager.provider_for(ModelRef(model='unknown-model')) is None


def test_manager_pins_a_model_to_the_named_provider() -> None:
    native = ProviderConfig(
        name='anthropic',
        adapter='anthropic',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8',),
    )
    gateway = ProviderConfig(
        name='bifrost',
        adapter='openai',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8',),
    )
    manager = LLMManager((native, gateway), HTTPClientManager(), logger=make_logger())

    pinned = ModelRef(model='claude-opus-4-8', provider='bifrost')
    assert manager.provider_for(pinned) is gateway
    assert isinstance(manager.client_for(pinned), OpenAIAdapter)


def test_manager_refuses_a_model_the_named_provider_does_not_offer() -> None:
    provider = ProviderConfig(
        name='openai',
        adapter='openai',
        api_key=SecretStr('secret'),
        models=('gpt-5',),
    )
    manager = LLMManager((provider,), HTTPClientManager(), logger=make_logger())

    assert manager.provider_for(ModelRef(model='gpt-4', provider='openai')) is None
    assert manager.provider_for(ModelRef(model='gpt-5', provider='ghost')) is None


def test_manager_refuses_a_bare_model_several_providers_claim() -> None:
    # NOTE: last-wins would silently pick one; ambiguity has to be named away
    native = ProviderConfig(
        name='anthropic',
        adapter='anthropic',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8',),
    )
    gateway = ProviderConfig(
        name='bifrost',
        adapter='openai',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8',),
    )
    manager = LLMManager((native, gateway), HTTPClientManager(), logger=make_logger())

    assert manager.provider_for(ModelRef(model='claude-opus-4-8')) is None


def test_manager_warns_once_about_every_contested_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    native = ProviderConfig(
        name='anthropic',
        adapter='anthropic',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8', 'claude-haiku-4-5'),
    )
    gateway = ProviderConfig(
        name='bifrost',
        adapter='openai',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8', 'gpt-5'),
    )

    with caplog.at_level('WARNING', logger='lib.llm.client'):
        LLMManager((native, gateway), HTTPClientManager(), logger=make_logger())

    assert len(caplog.records) == 1
    assert 'claude-opus-4-8' in caplog.text
    assert 'anthropic, bifrost' in caplog.text


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
    manager = LLMManager((anthropic, openai), HTTPClientManager(), logger=make_logger())

    claude = manager.client_for(ModelRef(model='claude-opus-4-8'))
    assert isinstance(claude, AnthropicAdapter)
    assert isinstance(manager.client_for(ModelRef(model='gpt-5')), OpenAIAdapter)
    assert manager.client_for(ModelRef(model='unknown-model')) is None


def test_client_for_caches_adapter_per_provider() -> None:
    provider = ProviderConfig(
        name='openai',
        adapter='openai',
        api_key=SecretStr('secret'),
        models=('gpt-5',),
    )
    manager = LLMManager((provider,), HTTPClientManager(), logger=make_logger())

    ref = ModelRef(model='gpt-5')
    assert manager.client_for(ref) is manager.client_for(ref)


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
    manager = LLMManager((provider,), HTTPClientManager(), logger=make_logger())

    with pytest.raises(ModuleNotFoundError, match=rf'huginn\[{provider.adapter}\]'):
        manager.client_for(ModelRef(model=model))
