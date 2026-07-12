from dataclasses import dataclass
from typing import cast

from fastapi import Request

from lib.container import Container


@dataclass(frozen=True)
class Provide[T]:
    binding: type[T]

    def __call__(self, request: Request) -> T:
        return cast(Container, request.app.state.container).resolve(self.binding)
