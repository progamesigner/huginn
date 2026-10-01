from pathlib import Path

import pytest

from app.core.providers.llm import LLMProvider
from app.providers.contributions import CoreContributions
from lib.llm import LLMContract, LLMManager, ModelRef
from tests.helpers.settings import app_with, make_settings

_CONFIG = """
providers:
  openai:
    adapter: openai
    api_key:
      value_from:
        env: OPENAI_API_KEY
    models:
    - gpt-5-mini
"""


def _write_config(path: Path) -> Path:
    path.write_text(_CONFIG)
    return path


def test_provider_builds_llm_manager_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')

    config = _write_config(tmp_path / 'huginn.yaml')
    container = app_with(make_settings(config_file=config)).build(
        LLMProvider(),
        CoreContributions(),
    )

    manager = container.resolve(LLMContract)

    assert isinstance(manager, LLMManager)
    provider = manager.provider_for(ModelRef(model='gpt-5-mini'))
    assert provider is not None
    assert provider.name == 'openai'
    assert provider.adapter == 'openai'
    assert provider.api_key.get_secret_value() == 'sk-test'
    assert manager.provider_for(ModelRef(model='unknown-model')) is None
