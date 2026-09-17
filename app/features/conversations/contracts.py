from abc import ABC, abstractmethod
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from .types import Blob

type BlobStoreFactory = Callable[[AsyncSession], BlobStoreContract]


class BlobStoreContract(ABC):
    @abstractmethod
    async def store(self, mime: str, data: bytes) -> Blob: ...


class ConversationControlContract(ABC):
    @abstractmethod
    async def compact(self, channel: str, chat: str) -> bool: ...

    @abstractmethod
    async def reset(self, channel: str, chat: str) -> bool: ...
