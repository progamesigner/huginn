from graphlib import CycleError, TopologicalSorter
from logging import getLogger
from typing import Self

from app.core.settings import Environment, HuginnSettings
from lib.container import Container, ProviderContract

logger = getLogger(__name__)


class BuildError(Exception):
    pass


def _reject_duplicate_providers(
    providers: tuple[ProviderContract, ...],
    items: dict[type[ProviderContract], ProviderContract],
) -> None:
    for provider in providers:
        if items[type(provider)] is not provider:
            raise BuildError(f'duplicate provider {type(provider).__name__}')


def _include_transitive_dependencies(
    providers: tuple[ProviderContract, ...],
    items: dict[type[ProviderContract], ProviderContract],
) -> None:
    pending = list(providers)

    while pending:
        provider = pending.pop()
        for dependency in provider.dependencies:
            if dependency in items:
                continue

            try:
                instance = dependency()
            except TypeError as error:
                raise BuildError(
                    f'provider {type(provider).__name__} depends on'
                    f' {dependency.__name__}, which needs explicit construction'
                    f' arguments and must be listed',
                ) from error

            items[dependency] = instance
            pending.append(instance)


def _sort_providers(
    items: dict[type[ProviderContract], ProviderContract],
) -> list[type[ProviderContract]]:
    graph = {key: set(provider.dependencies) for key, provider in items.items()}
    try:
        return list(TopologicalSorter(graph).static_order())
    except CycleError as error:
        names = ', '.join(node.__name__ for node in error.args[1])
        raise BuildError(f'provider dependency cycle: {names}') from error


class App(Container):
    def build(self, *providers: ProviderContract) -> Self:
        items = {type(provider): provider for provider in providers}

        _reject_duplicate_providers(providers, items)
        _include_transitive_dependencies(providers, items)

        for key in _sort_providers(items):
            logger.info('Registering provider %s', key.__name__)
            provider = items[key]
            provider(self)

        return self

    def running_in_mode(self, *modes: Environment) -> bool:
        return self.resolve(HuginnSettings).running_in_mode(*modes)
