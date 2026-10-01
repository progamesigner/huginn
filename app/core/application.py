from graphlib import CycleError, TopologicalSorter
from typing import Self

from app.core.settings import Environment, HuginnSettings
from lib.container import Container, ProviderContract
from lib.observability import Logger


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

            instance = dependency()
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

        order = _sort_providers(items)
        for key in order:
            items[key](self)

        # NOTE: the logger is itself wired by a provider, so the account comes after
        if self.bound(Logger):
            logger = self.resolve(Logger).bind(component='app')
            for key in order:
                logger.info('Registered provider %s', key.__name__)

        return self

    def running_in_mode(self, *modes: Environment) -> bool:
        return self.resolve(HuginnSettings).running_in_mode(*modes)
