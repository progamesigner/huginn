from typing import override

import pytest

from lib.container import BuildError, Container, ContainerContract, ProviderContract


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

    Container().build((Dependent(), Base()))

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
        Container().build((Orphan(),))


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
        Container().build((First(), Second()))


def test_build_rejects_duplicate_providers() -> None:
    class Twin(ProviderContract):
        @override
        def register(self, container: ContainerContract) -> None:
            pass

    with pytest.raises(BuildError, match='duplicate'):
        Container().build((Twin(), Twin()))


def test_build_logs_each_provider(caplog: pytest.LogCaptureFixture) -> None:
    class Settings(ProviderContract):
        @override
        def register(self, container: ContainerContract) -> None:
            pass

    with caplog.at_level('INFO', logger='lib.container.container'):
        Container().build((Settings(),))

    assert any('Settings' in message for message in caplog.messages)
