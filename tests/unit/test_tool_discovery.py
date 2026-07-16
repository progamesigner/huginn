import pytest

from app.core.tools import DuplicateToolError, build_registry, discover_tools


def test_discover_tools_finds_fixture_tool_via_auto_discovery() -> None:
    tools = discover_tools(('tests.fixtures.tool_packages.ok',))

    assert [tool.name for tool in tools] == ['greet']


def test_build_registry_fails_boot_on_duplicate_tool_name() -> None:
    with pytest.raises(DuplicateToolError, match='echo'):
        build_registry(('tests.fixtures.tool_packages.duplicate',))
