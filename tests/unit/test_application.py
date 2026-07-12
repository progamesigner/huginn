import logging
from typing import override

import pytest

from app.core.application import App
from app.core.exceptions import BuildError
from app.core.providers import LoggingProvider, SettingsProvider
from app.core.settings import AppSettings
from lib.container import ContainerContract, ProviderContract
from lib.observability import Logger


def test_build_registers_dependencies_first() -> None:
    journal: list[str] = []

    class Base(ProviderContract):
        @override
        def register(self, container: ContainerContract) -> None:
            journal.append('base')

    class Dependent(ProviderContract):
        dependencies = (Base,)

        @override
        def register(self, container: ContainerContract) -> None:
            journal.append('dependent')

    App().build(Dependent(), Base())

    assert journal.index('base') < journal.index('dependent')


def test_build_rejects_unknown_dependency() -> None:
    class Missing(ProviderContract):
        @override
        def register(self, container: ContainerContract) -> None:
            pass

    class Orphan(ProviderContract):
        dependencies = (Missing,)

        @override
        def register(self, container: ContainerContract) -> None:
            pass

    with pytest.raises(BuildError, match='unknown provider'):
        App().build(Orphan())


def test_build_rejects_dependency_cycle() -> None:
    class First(ProviderContract):
        @override
        def register(self, container: ContainerContract) -> None:
            pass

    class Second(ProviderContract):
        dependencies = (First,)

        @override
        def register(self, container: ContainerContract) -> None:
            pass

    First.dependencies = (Second,)

    with pytest.raises(BuildError, match='cycle'):
        App().build(First(), Second())


def test_build_rejects_duplicate_providers() -> None:
    class Twin(ProviderContract):
        @override
        def register(self, container: ContainerContract) -> None:
            pass

    with pytest.raises(BuildError, match='duplicate'):
        App().build(Twin(), Twin())


def test_build_logs_each_provider(caplog: pytest.LogCaptureFixture) -> None:
    class Settings(ProviderContract):
        @override
        def register(self, container: ContainerContract) -> None:
            pass

    with caplog.at_level('INFO', logger='app.core.application'):
        App().build(Settings())

    assert any('Settings' in message for message in caplog.messages)


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


def test_running_in_mode_matches_any_given_mode() -> None:
    application = App().build(SettingsProvider())

    with application.use(AppSettings, AppSettings(environment='staging')):
        assert application.running_in_mode('staging')
        assert application.running_in_mode('staging', 'prod')
        assert not application.running_in_mode('dev')
