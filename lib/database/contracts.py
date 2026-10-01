from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from sqlalchemy.ext.asyncio import AsyncSession


class DatabaseContract(ABC):
    @abstractmethod
    def session(
        self,
        alias: str | None = None,
    ) -> AbstractAsyncContextManager[AsyncSession]: ...

    # NOTE: commits on a clean exit and rolls back on an error
    @abstractmethod
    def transaction(
        self,
        alias: str | None = None,
    ) -> AbstractAsyncContextManager[AsyncSession]: ...

    @abstractmethod
    async def ping(self, alias: str | None = None) -> None: ...
