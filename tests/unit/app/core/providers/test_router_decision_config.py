from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.application import App
from app.core.config.registry import HuginnConfig
from app.core.providers.config import ConfigProvider
from app.features.agents.config import MAX_ROUTED_AGENTS, NO_AGENT
from app.features.agents.exceptions import (
    MissingModelError,
    ProviderModelMismatchError,
    ReservedAgentNameError,
    RouterShadowError,
    RouterThresholdError,
    TooManyRouterCandidatesError,
    UnknownProviderReferenceError,
)
from app.providers import HuginnProvider

_CONFIG = """
providers:
  llm:
    adapter: openai
    api_key: test
    models: [chat]
  jev:
    type: decision
    adapter: typesafe
    api_key: test
    models: [jev]
defaults:
  model: chat
agents:
  a:
    description: A
    instructions: A
api_keys:
  test:
    token: test
    agents: [a]
    router:
      engine: decision
      fallback: a
      allowed: [a]
      model: jev
      provider: jev
"""


@pytest.mark.parametrize(
    ('old', 'new', 'error'),
    [
        ('model: jev', 'model: missing', ProviderModelMismatchError),
        ('      model: jev', '', MissingModelError),
        ('provider: jev', 'provider: missing', UnknownProviderReferenceError),
    ],
)
def test_decision_model_must_resolve(
    tmp_path: Path,
    old: str,
    new: str,
    error: type[Exception],
) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text(_CONFIG.replace(old, new))
    container = App().build(ConfigProvider(path=path), HuginnProvider())
    with pytest.raises(error):
        container.resolve(HuginnConfig)


def test_valid_decision_model(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text(_CONFIG)
    container = App().build(ConfigProvider(path=path), HuginnProvider())
    container.resolve(HuginnConfig)


def test_channel_decision_config_uses_same_validation(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    prefix = _CONFIG.split('api_keys:')[0]
    path.write_text(
        prefix
        + """channels:
  test:
    kind: telegram
    token: test
    secret_token: test
    router:
      engine: decision
      fallback: a
      allowed: [a]
""",
    )
    container = App().build(ConfigProvider(path=path), HuginnProvider())
    with pytest.raises(MissingModelError, match=r'channels.test.router.model'):
        container.resolve(HuginnConfig)


def _resolve(tmp_path: Path, config: str) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text(config)
    App().build(ConfigProvider(path=path), HuginnProvider()).resolve(HuginnConfig)


@pytest.mark.parametrize('extra', [0, 1])
def test_decision_routing_caps_its_candidates(tmp_path: Path, extra: int) -> None:
    names = [f'agent{index}' for index in range(MAX_ROUTED_AGENTS + extra)]
    agents = ''.join(
        f'  {name}:\n    description: x\n    instructions: x\n' for name in names
    )
    config = _CONFIG.replace(
        '  a:\n    description: A\n    instructions: A\n',
        agents,
    ).replace('agents: [a]', f'agents: [{", ".join(names)}]')
    config = config.replace('fallback: a', 'fallback: agent0').replace(
        'allowed: [a]',
        f'allowed: [{", ".join(names)}]',
    )

    if extra:
        with pytest.raises(TooManyRouterCandidatesError, match='at most 254'):
            _resolve(tmp_path, config)
    else:
        _resolve(tmp_path, config)


def test_decision_routing_reserves_the_no_agent_name(tmp_path: Path) -> None:
    config = _CONFIG.replace(
        'agents:\n  a:\n',
        f'agents:\n  {NO_AGENT}:\n    description: N\n    instructions: N\n  a:\n',
    )

    with pytest.raises(ReservedAgentNameError, match=NO_AGENT):
        _resolve(tmp_path, config)


def test_thresholds_are_kept_for_decision_routing(tmp_path: Path) -> None:
    config = _CONFIG.replace(
        '      provider: jev\n',
        '      provider: jev\n      min_confidence: 0.6\n      min_margin: 0.1\n',
    )

    _resolve(tmp_path, config)


def test_thresholds_need_decision_routing(tmp_path: Path) -> None:
    config = _CONFIG.replace(
        '      engine: decision\n',
        '      min_confidence: 0.6\n',
    ).replace('      model: jev\n      provider: jev\n', '')

    with pytest.raises(RouterThresholdError, match='engine: decision'):
        _resolve(tmp_path, config)


def test_a_threshold_is_a_probability(tmp_path: Path) -> None:
    config = _CONFIG.replace(
        '      provider: jev\n',
        '      provider: jev\n      min_confidence: 1.5\n',
    )

    with pytest.raises(ValidationError, match='min_confidence'):
        _resolve(tmp_path, config)


def _shadowed(shadow: str) -> str:
    return _CONFIG.replace('      engine: decision\n', '').replace(
        '      model: jev\n      provider: jev\n',
        f'      model: chat\n      shadow:\n{shadow}',
    )


def test_an_llm_router_can_be_shadowed_by_a_decision_model(tmp_path: Path) -> None:
    _resolve(tmp_path, _shadowed('        model: jev\n        provider: jev\n'))


def test_the_shadow_model_must_resolve(tmp_path: Path) -> None:
    with pytest.raises(ProviderModelMismatchError, match=r'router\.shadow\.model'):
        _resolve(tmp_path, _shadowed('        model: missing\n'))


def test_a_decision_router_is_not_shadowed(tmp_path: Path) -> None:
    config = _CONFIG.replace(
        '      provider: jev\n',
        '      provider: jev\n      shadow:\n        model: jev\n',
    )

    with pytest.raises(RouterShadowError, match='only an llm router'):
        _resolve(tmp_path, config)
