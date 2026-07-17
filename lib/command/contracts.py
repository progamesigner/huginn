from abc import ABC, abstractmethod

from typer import Typer


class CommandContract(ABC):
    @property
    @abstractmethod
    def app(self) -> Typer: ...

    @abstractmethod
    def mount(self, name: str, commands: Typer) -> None: ...
