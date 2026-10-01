from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Annotated, cast, get_args, get_origin, override

from .contracts import ContainerContract, Disposable, Startable
from .exceptions import BindingError, ResolutionError

type Contributions = dict[type[object], list[Callable[[ContainerContract], object]]]
type Instances = dict[type[object], object]
type Registry = dict[type[object], Registration]
type Resolving = list[type[object]]


@dataclass(frozen=True)
class Registration:
    factory: Callable[[ContainerContract], object]
    shared: bool
    eager: bool


def wire[T](constructor: type[T]) -> Callable[[ContainerContract], T]:
    bindings = {}

    for parameter in signature(constructor).parameters.values():
        if parameter.kind in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL):
            continue

        if parameter.default is not Parameter.empty:
            continue

        if parameter.annotation is Parameter.empty:
            raise BindingError(
                f'{constructor.__qualname__} parameter {parameter.name!r}'
                f' has no annotation to resolve',
            )

        annotation = parameter.annotation
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]

        bindings[parameter.name] = annotation

    def factory(container: ContainerContract) -> T:
        kwargs = {
            name: container.resolve(binding) for name, binding in bindings.items()
        }
        return constructor(**kwargs)

    return factory


class Container(ContainerContract):
    contributions: Contributions
    instances: Instances
    overrides: Instances
    registry: Registry
    resolving: Resolving

    def __init__(self) -> None:
        self.contributions = {}
        self.instances = {}
        self.overrides = {}
        self.registry = {}
        self.resolving = []

    @override
    def bind[T](
        self,
        binding: type[T],
        *,
        constructor: type[T] | None = None,
        factory: Callable[[ContainerContract], T] | None = None,
        instance: T | None = None,
        shared: bool = False,
        eager: bool = False,
    ) -> None:
        name = binding.__qualname__

        if binding in self.registry:
            raise BindingError(f'{name} is already bound')

        if sum(source is not None for source in (constructor, factory, instance)) != 1:
            raise BindingError(f'{name} needs one of constructor/factory/instance')

        if constructor is not None:
            factory = wire(constructor)

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

        self.registry[binding] = Registration(
            factory,
            shared=shared,
            eager=eager,
        )

    @override
    def bound[T](self, binding: type[T]) -> bool:
        return binding in self.overrides or binding in self.registry

    @override
    def contribute[T](
        self,
        binding: type[T],
        factory: Callable[[ContainerContract], T],
    ) -> None:
        self.contributions.setdefault(binding, []).append(factory)

    @override
    def collect[T](self, binding: type[T]) -> tuple[T, ...]:
        return tuple(
            cast(T, factory(self)) for factory in self.contributions.get(binding, ())
        )

    @override
    def resolve[T](self, binding: type[T]) -> T:
        if binding in self.overrides:
            return cast(T, self.overrides[binding])

        if not (registration := self.registry.get(binding)):
            raise ResolutionError(f'{binding.__qualname__} is not bound')

        if binding in self.resolving:
            chain = ' -> '.join(t.__qualname__ for t in (*self.resolving, binding))
            raise ResolutionError(f'circular dependency: {chain}')

        self.resolving.append(binding)
        try:
            if not registration.shared:
                return cast(T, registration.factory(self))

            if binding not in self.instances:
                self.instances[binding] = registration.factory(self)

            return cast(T, self.instances[binding])
        finally:
            self.resolving.pop()

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

        container.contributions = {
            binding: list(factories)
            for binding, factories in self.contributions.items()
        }
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
        errors = []

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
