from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Annotated, cast, get_args, get_origin, override

from .contracts import ContainerContract, Disposable, Startable
from .exceptions import BindingError, ResolutionError

type Instances = dict[type[object], object]
type Registry = dict[type[object], Registration]
type Resolving = list[type[object]]


@dataclass(frozen=True)
class Registration:
    factory: Callable[[ContainerContract], object]
    shared: bool
    eager: bool
    scoped: bool = False


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
    instances: Instances
    overrides: Instances
    parent: Container | None
    registry: Registry
    resolving: Resolving

    def __init__(self) -> None:
        self.instances = {}
        self.overrides = {}
        self.parent = None
        self.registry = {}
        self.resolving = []

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
        constructor: type[T] | None = None,
        factory: Callable[[ContainerContract], T] | None = None,
        instance: T | None = None,
        shared: bool = False,
        scoped: bool = False,
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
            if scoped:
                raise BindingError(f'{name} cannot be scoped as a bare instance')

            self.instances[binding] = instance
            self.registry[binding] = Registration(
                lambda _: instance,
                shared=True,
                eager=eager,
            )
            return

        if eager and not shared:
            raise BindingError(f'{name} cannot be eager without shared')

        if scoped and (shared or eager):
            raise BindingError(f'{name} cannot be scoped and shared/eager')

        self.registry[binding] = Registration(
            factory,
            shared=shared,
            eager=eager,
            scoped=scoped,
        )

    @override
    def bound[T](self, binding: type[T]) -> bool:
        return self._override(binding) is not None or binding in self.registry

    @override
    def resolve[T](self, binding: type[T]) -> T:
        if (holder := self._override(binding)) is not None:
            return cast(T, holder.overrides[binding])

        if not (registration := self.registry.get(binding)):
            raise ResolutionError(f'{binding.__qualname__} is not bound')

        # NOTE: singletons always build and cache at the root, never in a child
        if registration.shared and (root := self._root()) is not self:
            return root.resolve(binding)

        if binding in self.resolving:
            chain = ' -> '.join(t.__qualname__ for t in (*self.resolving, binding))
            raise ResolutionError(f'circular dependency: {chain}')

        if registration.scoped and self.parent is None:
            chain = ' -> '.join(t.__qualname__ for t in (*self.resolving, binding))
            raise ResolutionError(
                f'{binding.__qualname__} is scoped and needs an active scope: {chain}',
            )

        self.resolving.append(binding)
        try:
            if not (registration.shared or registration.scoped):
                return cast(T, registration.factory(self))

            if binding not in self.instances:
                self.instances[binding] = registration.factory(self)

            return cast(T, self.instances[binding])
        finally:
            self.resolving.pop()

    @asynccontextmanager
    @override
    async def scope(self) -> AsyncIterator[ContainerContract]:
        child = Container()
        child.parent = self
        child.registry = self.registry

        try:
            yield child
        finally:
            await child.dispose()

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

    def _override(self, binding: type[object]) -> Container | None:
        node: Container | None = self
        while node is not None:
            if binding in node.overrides:
                return node
            node = node.parent
        return None

    def _root(self) -> Container:
        node = self
        while node.parent is not None:
            node = node.parent
        return node

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
