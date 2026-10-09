from datetime import timedelta
from hashlib import sha256
from typing import override
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.features.blobs.contracts import BlobServiceContract
from app.features.blobs.entities import Blob, BlobKind


class MemoryBlobs(BlobServiceContract):
    blobs: dict[UUID, Blob]
    grants: set[tuple[UUID, str]]
    ttls: dict[UUID, timedelta | None]

    def __init__(self) -> None:
        self.blobs = {}
        self.grants = set()
        self.ttls = {}

    @override
    async def store(
        self,
        transaction: AsyncSession,
        mime: str,
        data: bytes,
        *,
        kind: BlobKind,
        filename: str | None = None,
        ttl: timedelta | None = None,
    ) -> Blob:
        return await self.save(mime, data, kind=kind, filename=filename, ttl=ttl)

    @override
    async def save(
        self,
        mime: str,
        data: bytes,
        *,
        kind: BlobKind,
        filename: str | None = None,
        ttl: timedelta | None = None,
    ) -> Blob:
        blob = Blob(
            id=uuid4(),
            sha256=sha256(data).hexdigest(),
            mime=mime,
            size=len(data),
            kind=kind,
            filename=filename,
            data=data,
        )
        self.blobs[blob.id] = blob
        self.ttls[blob.id] = ttl
        return blob

    @override
    async def grant(self, id: UUID, owner: str) -> None:
        self.grants.add((id, owner))

    @override
    async def granted(self, id: UUID, owner: str) -> bool:
        return (id, owner) in self.grants

    @override
    async def load(self, id: UUID) -> Blob | None:
        return self.blobs.get(id)

    @override
    async def head(self, id: UUID) -> Blob | None:
        blob = self.blobs.get(id)
        return None if blob is None else blob.model_copy(update={'data': None})
