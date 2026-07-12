from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast, override

from .contracts import ContainerContract, Disposable, Startable
from .exceptions import BindingError, ResolutionError

type Instances = dict[type[object], object]
type Registry = dict[type[object], Registration]


@dataclass(frozen=True)
class Registration:
    factory: Callable[[ContainerContract], object]
    shared: bool
    eager: bool


class Container(ContainerContract):
    instances: Instances
    overrides: Instances
    registry: Registry

    def __init__(self) -> None:
        self.instances = cast(Instances, {})
        self.overrides = cast(Instances, {})
        self.registry = cast(Registry, {})

    @override
    def alias[T](self, binding: type[T], alias: type[T]) -> None:
        if alias in self.registry:
            raise BindingError(f'{alias.__qualname__} is already bound')

        if binding not in self.registry:
            raise BindingError(f'{binding.__qualname__} is not bound')

        self.registry[alias] = Registration(
            lambda container: container.resolve(binding),
            shared=False,
            eager=False,
        )

    @override
    def bind[T](
        self,
        binding: type[T],
        *,
        instance: T | None = None,
        factory: Callable[[ContainerContract], T] | None = None,
        shared: bool = False,
        eager: bool = False,
    ) -> None:
        name = binding.__qualname__

        if binding in self.registry:
            raise BindingError(f'{name} is already bound')

        if not (instance is None) ^ (factory is None):
            raise BindingError(f'{name} needs exactly one of instance or factory')

        if factory is None:
            self.instances[binding] = instance
            self.registry[binding] = Registration(
                lambda _: instance,
                shared=True,
                eager=eager,
            )
            return

        if eager and not shared:
            raise BindingError(f'{name} cannot be eager without shared')

        self.registry[binding] = Registration(factory, shared=shared, eager=eager)

    @override
    def bound[T](self, binding: type[T]) -> bool:
        return binding in self.overrides or binding in self.registry

    @override
    def resolve[T](self, binding: type[T]) -> T:
        if binding in self.overrides:
            return cast(T, self.overrides[binding])

        if not (registration := self.registry.get(binding)):
            raise ResolutionError(f'{binding.__qualname__} is not bound')

        if not registration.shared:
            return cast(T, registration.factory(self))

        if binding not in self.instances:
            self.instances[binding] = registration.factory(self)

        return cast(T, self.instances[binding])

    @contextmanager
    @override
    def use[T](self, binding: type[T], value: T) -> Iterator[T]:
        # NOTE: for tests; do not overlap use() on the same binding across tasks
        replaced = binding in self.overrides
        previous = self.overrides.get(binding)

        self.overrides[binding] = value
        try:
            yield value
        finally:
            if replaced:
                self.overrides[binding] = previous
            else:
                del self.overrides[binding]

    @override
    def to[C: ContainerContract](self, constructor: type[C]) -> C:
        container = constructor()
        if not isinstance(container, Container):
            raise BindingError(f'{constructor.__qualname__} is not a Container')

        container.registry = dict(self.registry)
        return container

    @override
    async def start(self) -> None:
        try:
            for binding, registration in self.registry.items():
                if not registration.eager:
                    continue

                instance = self.resolve(binding)
                if isinstance(instance, Startable):
                    await instance.start()
        except BaseException:
            await self.dispose()
            raise

    @override
    async def dispose(self) -> None:
        errors = cast(list[Exception], [])

        # NOTE: snapshot; a disposing manager may resolve and grow instances mid-loop
        for instance in reversed(list(self.instances.values())):
            if not isinstance(instance, Disposable):
                continue

            try:
                await instance.dispose()
            except Exception as error:
                errors.append(error)

        self.instances.clear()

        if len(errors) == 1:
            raise errors[0]

        if errors:
            raise ExceptionGroup('errors while disposing container', errors)
