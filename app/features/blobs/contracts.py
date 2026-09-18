from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
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
    async def referenced(self, ids: Collection[UUID]) -> frozenset[UUID]: ...


type BlobReferenceFactory = Callable[[AsyncSession], BlobReferenceContract]


class BlobReferenceRegistry:
    factories: list[BlobReferenceFactory]

    def __init__(self) -> None:
        self.factories = []

    def register(self, factory: BlobReferenceFactory) -> None:
        self.factories.append(factory)

    def for_session(self, session: AsyncSession) -> tuple[BlobReferenceContract, ...]:
        return tuple(factory(session) for factory in self.factories)
