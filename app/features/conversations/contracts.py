from abc import ABC, abstractmethod
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from .types import Blob

type BlobStoreFactory = Callable[[AsyncSession], BlobStoreContract]


class BlobStoreContract(ABC):
    @abstractmethod
    async def store(self, mime: str, data: bytes) -> Blob: ...
