from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.application import App
from app.core.config.decision import DecisionProviderConfig
from app.core.config.llm import ProviderConfig
from app.core.config.providers import ProviderConfigList
from app.core.providers.config import ConfigProvider
from app.core.providers.decision import DecisionProvider
from lib.decision import DecisionContract, DecisionRef


@pytest.mark.parametrize('configured', [False, True])
def test_decision_container(tmp_path: Path, configured: bool) -> None:
    path = tmp_path / 'huginn.yaml'
    path.write_text(
        """providers:
  direct:
    type: decision
    adapter: typesafe
    api_key: test-key
    models: [jev]
"""
        if configured
        else '{}',
    )
    container = App().build(ConfigProvider(path=path), DecisionProvider())
    manager = container.resolve(DecisionContract)
    provider = manager.provider_for(DecisionRef(model='jev'))
    assert (provider is not None) == configured
    if provider is not None:
        assert provider.timeout == 3.0
    assert manager.client_for(DecisionRef(model='unknown')) is None


@pytest.mark.parametrize(
    'changes',
    [
        {'models': []},
        {'models': None},
        {'timeout': 0},
        {'timeout': -1},
        {'timeout': float('inf')},
        {'adapter': 'unknown'},
        {'adapter': 'bifrost'},
    ],
)
def test_invalid_decision_config(changes: dict[str, object]) -> None:
    values = {
        'type': 'decision',
        'adapter': 'typesafe',
        'api_key': 'test',
        'models': ('jev',),
    }
    values.update(changes)
    with pytest.raises(ValidationError):
        ProviderConfigList.model_validate({'gateway': values})


def test_a_provider_without_a_type_is_an_llm_provider() -> None:
    providers = ProviderConfigList.model_validate(
        {
            'chat': {'adapter': 'openai', 'api_key': 'test', 'models': ('gpt',)},
            'jev': {
                'type': 'decision',
                'adapter': 'bifrost',
                'api_key': 'test',
                'models': ('typesafe/jev-1.13.0',),
                'url': 'http://bifrost:8080/v1/decisions',
            },
        },
    ).root

    assert isinstance(providers['chat'], ProviderConfig)
    assert isinstance(providers['jev'], DecisionProviderConfig)


def test_an_llm_only_field_is_refused_on_a_decision_provider() -> None:
    with pytest.raises(ValidationError, match='cache_control'):
        ProviderConfigList.model_validate(
            {
                'jev': {
                    'type': 'decision',
                    'adapter': 'typesafe',
                    'api_key': 'test',
                    'models': ('jev',),
                    'cache_control': ('jev',),
                },
            },
        )
