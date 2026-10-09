from collections.abc import Sequence
from datetime import datetime, timedelta
from hashlib import sha256
from typing import cast
from uuid import UUID

from sqlalchemy import case, delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.features.blobs.contracts import BlobReferenceSource
from app.features.blobs.entities import Blob, BlobKind
from app.features.blobs.exceptions import BlobError
from app.features.blobs.sources.sqlalchemy import BlobGrantRecord, BlobRecord
from lib.guid import GUIDContract
from lib.temporal import ClockContract


def _to_blob(record: BlobRecord, *, data: bytes | None = None) -> Blob:
    return Blob(
        id=record.id,
        sha256=record.sha256,
        mime=record.mime,
        size=record.size,
        kind=cast(BlobKind, record.kind),
        filename=record.filename,
        expires_at=record.expires_at,
        data=data,
    )


class BlobRepository:
    def __init__(
        self,
        session: AsyncSession,
        guid: GUIDContract,
        clock: ClockContract,
    ) -> None:
        self.session = session
        self.guid = guid
        self.clock = clock

    async def store(
        self,
        mime: str,
        data: bytes,
        *,
        kind: BlobKind,
        filename: str | None = None,
        ttl: timedelta | None = None,
    ) -> Blob:
        digest = sha256(data).hexdigest()
        expires_at = None if ttl is None else self.clock.now() + ttl
        retained_until = (
            None
            if expires_at is None
            else case(
                (BlobRecord.expires_at.is_(None), None),
                else_=func.greatest(BlobRecord.expires_at, expires_at),
            )
        )
        inserted = (
            insert(BlobRecord)
            .values(
                id=self.guid.generate(),
                sha256=digest,
                mime=mime,
                size=len(data),
                data=data,
                kind=kind,
                filename=filename,
                expires_at=expires_at,
            )
            .on_conflict_do_update(
                index_elements=[BlobRecord.sha256],
                set_={
                    'expires_at': retained_until,
                    'filename': func.coalesce(BlobRecord.filename, filename),
                },
            )
            .returning(BlobRecord)
        )
        record = (await self.session.execute(inserted)).scalar_one()
        return _to_blob(record)

    async def load(self, id: UUID) -> Blob | None:
        record = await self.session.scalar(
            select(BlobRecord).where(BlobRecord.id == id),
        )
        return None if record is None else _to_blob(record, data=record.data)

    async def head(self, id: UUID) -> Blob | None:
        record = await self.session.scalar(
            select(BlobRecord)
            .options(defer(BlobRecord.data))
            .where(BlobRecord.id == id),
        )
        return None if record is None else _to_blob(record)

    async def grant(self, id: UUID, owner: str) -> None:
        await self.session.execute(
            insert(BlobGrantRecord)
            .values(id=self.guid.generate(), blob_id=id, owner=owner)
            .on_conflict_do_nothing(
                index_elements=[BlobGrantRecord.blob_id, BlobGrantRecord.owner],
            ),
        )

    async def granted(self, id: UUID, owner: str) -> bool:
        found = await self.session.scalar(
            select(BlobGrantRecord.id).where(
                BlobGrantRecord.blob_id == id,
                BlobGrantRecord.owner == owner,
            ),
        )
        return found is not None

    async def prune(
        self,
        before: datetime,
        references: Sequence[BlobReferenceSource],
        *,
        dry_run: bool = False,
    ) -> int:
        if not references:
            raise BlobError('no blob reference source is registered; refusing to prune')

        expired = frozenset(
            await self.session.scalars(
                select(BlobRecord.id).where(BlobRecord.expires_at < before),
            ),
        )
        kept = frozenset[UUID]().union(
            *[
                await reference.referenced(self.session, expired)
                for reference in references
            ],
        )
        doomed = expired - kept
        if doomed and not dry_run:
            await self.session.execute(
                delete(BlobRecord).where(BlobRecord.id.in_(doomed)),
            )
        return len(doomed)
