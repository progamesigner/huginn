from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from sqlalchemy.ext.asyncio import AsyncSession


class DatabaseContract(ABC):
    @abstractmethod
    def session(self) -> AbstractAsyncContextManager[AsyncSession]: ...
