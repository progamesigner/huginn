from abc import ABC, abstractmethod

from fastapi import APIRouter
from starlette.types import ASGIApp


class RouterContract(ABC):
    @property
    @abstractmethod
    def app(self) -> ASGIApp: ...

    @abstractmethod
    def include(self, router: APIRouter) -> None: ...
