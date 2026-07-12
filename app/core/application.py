from graphlib import CycleError, TopologicalSorter
from logging import getLogger
from typing import Self

from app.core.exceptions import BuildError
from lib.container import Container, ProviderContract

logger = getLogger(__name__)


class App(Container):
    def build(self, *providers: ProviderContract) -> Self:
        items = {type(provider): provider for provider in providers}
        graph = {type(provider): set(provider.dependencies) for provider in providers}

        for provider in providers:
            if items[type(provider)] is not provider:
                raise BuildError(f'duplicate provider {type(provider).__name__}')

        for provider in providers:
            for dependency in provider.dependencies:
                if dependency not in items:
                    raise BuildError(
                        f'provider {type(provider).__name__} depends on'
                        f' unknown provider {dependency.__name__}'
                    )

        try:
            keys = list(TopologicalSorter(graph).static_order())
        except CycleError as error:
            names = ', '.join(node.__name__ for node in error.args[1])
            raise BuildError(f'provider dependency cycle: {names}') from error

        for key in keys:
            logger.info('Registering provider %s', key.__name__)
            items[key].register(self)

        return self
