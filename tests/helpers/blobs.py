from datetime import timedelta
from hashlib import sha256
from typing import override
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.features.blobs.contracts import BlobServiceContract
from app.features.blobs.entities import Blob, BlobGrant, BlobKind


class MemoryBlobs(BlobServiceContract):
    blobs: dict[UUID, Blob]
    grants: dict[tuple[UUID, str], str | None]
    ttls: dict[UUID, timedelta | None]

    def __init__(self) -> None:
        self.blobs = {}
        self.grants = {}
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
        # NOTE: the same bytes stay one blob, as they do in the database
        digest = sha256(data).hexdigest()
        for kept in self.blobs.values():
            if kept.sha256 == digest:
                return kept

        blob = Blob(
            id=uuid4(),
            sha256=digest,
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
    async def grant(
        self,
        id: UUID,
        owner: str,
        *,
        filename: str | None = None,
    ) -> None:
        self.grants[id, owner] = filename or self.grants.get((id, owner))

    @override
    async def granted(self, id: UUID, owner: str) -> BlobGrant | None:
        if (id, owner) not in self.grants:
            return None

        return BlobGrant(blob_id=id, owner=owner, filename=self.grants[id, owner])

    @override
    async def load(self, id: UUID) -> Blob | None:
        return self.blobs.get(id)

    @override
    async def head(self, id: UUID) -> Blob | None:
        blob = self.blobs.get(id)
        return None if blob is None else blob.model_copy(update={'data': None})
