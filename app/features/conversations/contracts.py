from abc import ABC, abstractmethod

from .types import Blob


class BlobStoreContract(ABC):
    @abstractmethod
    async def store(self, mime: str, data: bytes) -> Blob: ...
