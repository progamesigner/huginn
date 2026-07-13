from graphlib import CycleError, TopologicalSorter
from logging import getLogger
from typing import ClassVar, Self

from app.core.exceptions import BuildError
from app.core.providers.logging import LoggingProvider
from app.core.providers.settings import SettingsProvider
from app.core.settings import AppSettings, Environment
from lib.container import Container, ProviderContract

logger = getLogger(__name__)


class App(Container):
    providers: ClassVar[tuple[ProviderContract, ...]] = (
        SettingsProvider(),
        LoggingProvider(),
    )

    def build(self, *providers: ProviderContract) -> Self:
        items = {type(provider): provider for provider in providers}

        for provider in providers:
            if items[type(provider)] is not provider:
                raise BuildError(f'duplicate provider {type(provider).__name__}')

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
                        f' arguments and must be listed'
                    ) from error

                items[dependency] = instance
                pending.append(instance)

        graph = {key: set(provider.dependencies) for key, provider in items.items()}
        try:
            keys = list(TopologicalSorter(graph).static_order())
        except CycleError as error:
            names = ', '.join(node.__name__ for node in error.args[1])
            raise BuildError(f'provider dependency cycle: {names}') from error

        for key in keys:
            logger.info('Registering provider %s', key.__name__)
            items[key].register(self)

        return self

    def running_in_mode(self, *modes: Environment) -> bool:
        return self.resolve(AppSettings).running_in_mode(*modes)
