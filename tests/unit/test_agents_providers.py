from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.application import App
from app.core.config import SectionRegistry
from app.exts.felo.client import FeloClient
from app.features.agents.providers import (
    AgentsConfigProvider,
    ToolRegistryProvider,
    WebSearchToolProvider,
)
from app.features.agents.tools import ToolRegistry


@pytest.fixture(autouse=True)
def felo_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EXT_FELO_API_KEY', 'felo-key')
    monkeypatch.delenv('EXT_FELO_BASE_URL', raising=False)


async def test_tool_registry_provider_binds_registry() -> None:
    app = App().build(ToolRegistryProvider())

    assert app.bound(ToolRegistry)


async def test_websearch_provider_registers_declaration() -> None:
    app = App().build(WebSearchToolProvider())

    try:
        assert 'websearch' in app.resolve(ToolRegistry).tools
    finally:
        await app.dispose()


async def test_registration_stays_lazy_without_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('EXT_FELO_API_KEY', raising=False)

    app = App().build(WebSearchToolProvider())

    try:
        assert 'websearch' in app.resolve(ToolRegistry).tools
        with pytest.raises(ValidationError):
            app.resolve(FeloClient)
    finally:
        await app.dispose()


async def test_config_provider_contributes_sections() -> None:
    app = App().build(AgentsConfigProvider())

    keys = {spec.key for spec in app.resolve(SectionRegistry)}

    assert keys == {'agents', 'defaults', 'mcp_servers', 'channels'}
