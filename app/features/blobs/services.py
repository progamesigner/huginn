from datetime import timedelta
from typing import override
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from lib.database import DatabaseContract
from lib.guid import GUIDContract
from lib.temporal import ClockContract

from .contracts import BlobServiceContract
from .entities import Blob, BlobKind
from .repositories import BlobRepository


class BlobService(BlobServiceContract):
    def __init__(
        self,
        database: DatabaseContract,
        guid: GUIDContract,
        clock: ClockContract,
    ) -> None:
        self.database = database
        self.guid = guid
        self.clock = clock

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
        return await BlobRepository(transaction, self.guid, self.clock).store(
            mime,
            data,
            kind=kind,
            filename=filename,
            ttl=ttl,
        )

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
        async with self.database.transaction() as transaction:
            return await self.store(
                transaction,
                mime,
                data,
                kind=kind,
                filename=filename,
                ttl=ttl,
            )

    # NOTE: a read needs no transaction of the caller's, so it opens its own session
    @override
    async def load(self, id: UUID) -> Blob | None:
        async with self.database.session() as session:
            return await BlobRepository(session, self.guid, self.clock).load(id)

    @override
    async def head(self, id: UUID) -> Blob | None:
        async with self.database.session() as session:
            return await BlobRepository(session, self.guid, self.clock).head(id)
