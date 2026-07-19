from app.core.application import App
from app.core.config.registry import SectionRegistry
from app.features.agents.providers import AgentsConfigProvider
from app.features.tools.providers import (
    ToolRegistryProvider,
    ToolsConfigProvider,
    WebSearchToolProvider,
)
from app.features.tools.registry import ToolRegistry
from app.features.tools.repositories import WebSearchRepository


def test_tool_registry_provider_binds_registry() -> None:
    app = App().build(ToolRegistryProvider())

    assert app.bound(ToolRegistry)


def test_websearch_provider_binds_repository() -> None:
    app = App().build(WebSearchToolProvider())

    assert app.bound(WebSearchRepository)


def test_websearch_provider_declares_kind() -> None:
    app = App().build(WebSearchToolProvider())

    assert 'websearch' in app.resolve(ToolRegistry).tools


def test_tools_config_provider_contributes_tools_section() -> None:
    app = App().build(ToolsConfigProvider())

    keys = {spec.key for spec in app.resolve(SectionRegistry)}

    assert keys == {'tools'}


def test_agents_config_provider_contributes_all_sections() -> None:
    app = App().build(AgentsConfigProvider())

    keys = {spec.key for spec in app.resolve(SectionRegistry)}

    assert keys == {'agents', 'defaults', 'mcp_servers', 'channels', 'tools'}
