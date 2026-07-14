from collections.abc import Callable
from dataclasses import dataclass

type Result[T, E: Exception] = Ok[T] | Err[E]


@dataclass(frozen=True)
class Ok[T]:
    value: T

    @property
    def ok(self) -> bool:
        return True

    @property
    def err(self) -> bool:
        return False

    def map[U](self, fn: Callable[[T], U]) -> Ok[U]:
        return Ok(fn(self.value))

    def and_then[U, E: Exception](
        self,
        fn: Callable[[T], Result[U, E]],
    ) -> Result[U, E]:
        return fn(self.value)

    def unwrap_or(self, default: T) -> T:
        return self.value


@dataclass(frozen=True)
class Err[E: Exception]:
    error: E

    @property
    def ok(self) -> bool:
        return False

    @property
    def err(self) -> bool:
        return True

    def map[T, U](self, fn: Callable[[T], U]) -> Err[E]:
        return self

    def and_then[T, U](self, fn: Callable[[T], Result[U, E]]) -> Err[E]:
        return self

    def unwrap_or[T](self, default: T) -> T:
        return default
