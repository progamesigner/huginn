from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from sqlalchemy.ext.asyncio import AsyncSession

from .session import ScopedSession


class DatabaseContract(ABC):
    @abstractmethod
    def scoped(self, instance: str | None = None) -> ScopedSession: ...

    @abstractmethod
    def session(
        self,
        instance: str | None = None,
    ) -> AbstractAsyncContextManager[AsyncSession]: ...

    @abstractmethod
    def url(self, instance: str | None = None) -> str: ...
