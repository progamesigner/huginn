from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from datetime import timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from .entities import Blob, BlobGrant, BlobKind


class BlobServiceContract(ABC):
    # NOTE: a write joins the caller's transaction, so a blob lands with what names it
    @abstractmethod
    async def store(
        self,
        transaction: AsyncSession,
        mime: str,
        data: bytes,
        *,
        kind: BlobKind,
        filename: str | None = None,
        ttl: timedelta | None = None,
    ) -> Blob: ...

    # NOTE: for bytes nothing in a transaction names yet -- an upload or a generated
    # file -- so the blob commits alone and lapses on `ttl` unless something adopts it
    @abstractmethod
    async def save(
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

    # NOTE: a grant names who may fetch a blob back; it never keeps one from lapsing
    @abstractmethod
    async def grant(
        self,
        id: UUID,
        owner: str,
        *,
        filename: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def granted(self, id: UUID, owner: str) -> BlobGrant | None: ...


class BlobReferenceSource(ABC):
    @abstractmethod
    async def referenced(
        self,
        session: AsyncSession,
        ids: Collection[UUID],
    ) -> frozenset[UUID]: ...


class BlobReferenceRegistry:
    references: tuple[BlobReferenceSource, ...]

    def __init__(self, references: Iterable[BlobReferenceSource] = ()) -> None:
        self.references = tuple(references)
