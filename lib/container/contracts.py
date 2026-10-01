from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import TracebackType
from typing import ClassVar, Protocol, Self, runtime_checkable


@runtime_checkable
class Startable(Protocol):
    async def start(self) -> None: ...


@runtime_checkable
class Disposable(Protocol):
    async def dispose(self) -> None: ...


class ContainerContract(ABC):
    @abstractmethod
    def bind[T](
        self,
        binding: type[T],
        *,
        constructor: type[T] | None = None,
        factory: Callable[[ContainerContract], T] | None = None,
        instance: T | None = None,
        shared: bool = False,
        eager: bool = False,
    ) -> None: ...

    @abstractmethod
    def bound[T](self, binding: type[T]) -> bool: ...

    @abstractmethod
    def resolve[T](self, binding: type[T]) -> T: ...

    @abstractmethod
    def to[C: ContainerContract](self, constructor: type[C]) -> C: ...

    @abstractmethod
    def use[T](self, binding: type[T], value: T) -> AbstractContextManager[T]: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def dispose(self) -> None: ...

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.dispose()


class ProviderContract(ABC):
    dependencies: ClassVar[tuple[type[ProviderContract], ...]] = ()

    @abstractmethod
    def __call__(self, container: ContainerContract) -> None: ...
