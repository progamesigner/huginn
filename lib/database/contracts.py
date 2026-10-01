from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from sqlalchemy.ext.asyncio import AsyncSession

from .session import ScopedSession


class DatabaseContract(ABC):
    @abstractmethod
    def scoped(self, alias: str | None = None) -> ScopedSession: ...

    @abstractmethod
    def session(
        self,
        alias: str | None = None,
    ) -> AbstractAsyncContextManager[AsyncSession]: ...

    @abstractmethod
    def url(self, alias: str | None = None) -> str: ...
