from abc import ABC, abstractmethod
from typing import Protocol

from typer import Typer


class CommandContract(ABC):
    @property
    @abstractmethod
    def app(self) -> Typer: ...

    @abstractmethod
    def mount(self, name: str, commands: Typer) -> None: ...


# NOTE: structural so any container fits without lib/command importing one
class Resolver(Protocol):
    def resolve[T](self, binding: type[T]) -> T: ...
