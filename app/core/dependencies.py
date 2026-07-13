from dataclasses import dataclass
from typing import cast

from lib.container import Container
from lib.router import Request


@dataclass(frozen=True)
class Provide[T]:
    binding: type[T]

    def __call__(self, request: Request) -> T:
        return cast(Container, request.app.state.container).resolve(self.binding)
