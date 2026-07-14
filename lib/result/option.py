from collections.abc import Callable
from dataclasses import dataclass

from .result import Err, Ok

type Option[T] = Some[T] | Nothing


@dataclass(frozen=True)
class Some[T]:
    value: T

    @property
    def some(self) -> bool:
        return True

    @property
    def none(self) -> bool:
        return False

    def map[U](self, fn: Callable[[T], U]) -> Some[U]:
        return Some(fn(self.value))

    def and_then[U](self, fn: Callable[[T], Option[U]]) -> Option[U]:
        return fn(self.value)

    def unwrap_or(self, default: T) -> T:
        return self.value

    def ok_or[E: Exception](self, error: E) -> Ok[T]:
        return Ok(self.value)


@dataclass(frozen=True)
class Nothing:
    @property
    def some(self) -> bool:
        return False

    @property
    def none(self) -> bool:
        return True

    def map[T, U](self, fn: Callable[[T], U]) -> Nothing:
        return self

    def and_then[T, U](self, fn: Callable[[T], Option[U]]) -> Nothing:
        return self

    def unwrap_or[T](self, default: T) -> T:
        return default

    def ok_or[E: Exception](self, error: E) -> Err[E]:
        return Err(error)
