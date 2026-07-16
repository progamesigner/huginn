import logging
from typing import override

import pytest

from app.core.application import App
from app.core.exceptions import BuildError
from app.core.providers.llm import LLMProvider
from app.core.providers.logging import LoggingProvider
from app.core.providers.settings import SettingsProvider
from app.core.providers.tools import ToolRegistryProvider
from app.core.settings import AppSettings
from app.core.tools import DuplicateToolError, ToolRegistry
from lib.container import ContainerContract, ProviderContract
from lib.llm import LLMManager
from lib.observability import Logger


def test_build_registers_dependencies_first() -> None:
    journal: list[str] = []

    class Base(ProviderContract):
        @override
        def __call__(self, container: ContainerContract) -> None:
            journal.append('base')

    class Dependent(ProviderContract):
        dependencies = (Base,)

        @override
        def __call__(self, container: ContainerContract) -> None:
            journal.append('dependent')

    App().build(Dependent(), Base())

    assert journal.index('base') < journal.index('dependent')


def test_build_auto_includes_declared_dependency() -> None:
    journal: list[str] = []

    class Base(ProviderContract):
        @override
        def __call__(self, container: ContainerContract) -> None:
            journal.append('base')

    class Dependent(ProviderContract):
        dependencies = (Base,)

        @override
        def __call__(self, container: ContainerContract) -> None:
            journal.append('dependent')

    App().build(Dependent())

    assert journal == ['base', 'dependent']


def test_build_auto_includes_transitive_dependencies() -> None:
    journal: list[str] = []

    class Root(ProviderContract):
        @override
        def __call__(self, container: ContainerContract) -> None:
            journal.append('root')

    class Middle(ProviderContract):
        dependencies = (Root,)

        @override
        def __call__(self, container: ContainerContract) -> None:
            journal.append('middle')

    class Leaf(ProviderContract):
        dependencies = (Middle,)

        @override
        def __call__(self, container: ContainerContract) -> None:
            journal.append('leaf')

    App().build(Leaf())

    assert journal == ['root', 'middle', 'leaf']


def test_build_prefers_explicit_instance_over_auto_include() -> None:
    seen: list[object] = []

    class Base(ProviderContract):
        @override
        def __call__(self, container: ContainerContract) -> None:
            seen.append(self)

    class Dependent(ProviderContract):
        dependencies = (Base,)

        @override
        def __call__(self, container: ContainerContract) -> None:
            pass

    explicit = Base()
    App().build(explicit, Dependent())

    assert seen == [explicit]


def test_build_reports_dependency_needing_construction_arguments() -> None:
    class NeedsArgs(ProviderContract):
        def __init__(self, value: str) -> None:
            self.value = value

        @override
        def __call__(self, container: ContainerContract) -> None:
            pass

    class Dependent(ProviderContract):
        dependencies = (NeedsArgs,)

        @override
        def __call__(self, container: ContainerContract) -> None:
            pass

    with pytest.raises(BuildError, match='NeedsArgs'):
        App().build(Dependent())


def test_build_rejects_dependency_cycle() -> None:
    class First(ProviderContract):
        @override
        def __call__(self, container: ContainerContract) -> None:
            pass

    class Second(ProviderContract):
        dependencies = (First,)

        @override
        def __call__(self, container: ContainerContract) -> None:
            pass

    First.dependencies = (Second,)

    with pytest.raises(BuildError, match='cycle'):
        App().build(First(), Second())


def test_build_rejects_duplicate_providers() -> None:
    class Twin(ProviderContract):
        @override
        def __call__(self, container: ContainerContract) -> None:
            pass

    with pytest.raises(BuildError, match='duplicate'):
        App().build(Twin(), Twin())


def test_build_logs_each_provider(caplog: pytest.LogCaptureFixture) -> None:
    class Settings(ProviderContract):
        @override
        def __call__(self, container: ContainerContract) -> None:
            pass

    with caplog.at_level('INFO', logger='app.core.application'):
        App().build(Settings())

    assert any('Settings' in message for message in caplog.messages)


def test_provider_is_callable() -> None:
    provider = SettingsProvider()

    assert isinstance(provider, ProviderContract)
    assert callable(provider)


def test_build_wires_app_providers() -> None:
    application = App().build(SettingsProvider(), LoggingProvider())

    assert application.bound(AppSettings)
    assert application.bound(Logger)


def test_resolved_logger_is_bound_logger() -> None:
    application = App().build(SettingsProvider(), LoggingProvider())

    assert isinstance(application.resolve(Logger).bind(), Logger)


def test_build_routes_uvicorn_loggers_through_root() -> None:
    App().build(SettingsProvider(), LoggingProvider())

    for name in ('uvicorn', 'uvicorn.error'):
        assert logging.getLogger(name).handlers == []
        assert logging.getLogger(name).propagate

    access = logging.getLogger('uvicorn.access')
    assert access.handlers == []
    assert not access.propagate


def test_build_wires_llm_provider() -> None:
    application = App().build(SettingsProvider(), LLMProvider())

    assert application.bound(LLMManager)


async def test_llm_manager_disposes_cleanly_through_container() -> None:
    application = App().build(SettingsProvider(), LLMProvider())
    application.resolve(LLMManager)

    await application.dispose()


def test_build_wires_tool_registry_provider() -> None:
    application = App().build(ToolRegistryProvider())

    assert application.bound(ToolRegistry)


def test_build_fails_when_discovered_tools_collide() -> None:
    provider = ToolRegistryProvider(
        packages=('tests.fixtures.tool_packages.duplicate',)
    )

    with pytest.raises(DuplicateToolError, match='echo'):
        App().build(provider)


def test_running_in_mode_matches_any_given_mode() -> None:
    application = App().build(SettingsProvider())

    with application.use(AppSettings, AppSettings(environment='staging')):
        assert application.running_in_mode('staging')
        assert application.running_in_mode('staging', 'prod')
        assert not application.running_in_mode('dev')
