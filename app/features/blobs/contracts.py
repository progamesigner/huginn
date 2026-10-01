from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from datetime import timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from .types import Blob, BlobKind


class BlobStoreContract(ABC):
    @abstractmethod
    async def store(
        self,
        mime: str,
        data: bytes,
        *,
        kind: BlobKind,
        filename: str | None = None,
        ttl: timedelta | None = None,
    ) -> Blob: ...

    @abstractmethod
    async def load(self, id: UUID) -> Blob | None: ...

    @abstractmethod
    async def head(self, id: UUID) -> Blob | None: ...


class BlobStoreFactoryContract(ABC):
    @abstractmethod
    def for_session(self, session: AsyncSession) -> BlobStoreContract: ...


class BlobReferenceContract(ABC):
    @abstractmethod
    async def referenced(
        self,
        session: AsyncSession,
        ids: Collection[UUID],
    ) -> frozenset[UUID]: ...


class BlobReferenceRegistry:
    references: tuple[BlobReferenceContract, ...]

    def __init__(self, references: Iterable[BlobReferenceContract] = ()) -> None:
        self.references = tuple(references)
