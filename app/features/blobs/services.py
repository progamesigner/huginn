from datetime import timedelta
from typing import override
from uuid import UUID

from lib.database import DatabaseContract
from lib.guid import GUIDContract
from lib.temporal import ClockContract

from .contracts import BlobStoreContract
from .entities import Blob, BlobKind
from .repositories import BlobRepository


# NOTE: each call runs in its own session, for a caller that holds no transaction
class BlobService(BlobStoreContract):
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
        mime: str,
        data: bytes,
        *,
        kind: BlobKind,
        filename: str | None = None,
        ttl: timedelta | None = None,
    ) -> Blob:
        async with self.database.transaction() as transaction:
            return await BlobRepository(transaction, self.guid, self.clock).store(
                mime,
                data,
                kind=kind,
                filename=filename,
                ttl=ttl,
            )

    @override
    async def load(self, id: UUID) -> Blob | None:
        async with self.database.session() as session:
            return await BlobRepository(session, self.guid, self.clock).load(id)

    @override
    async def head(self, id: UUID) -> Blob | None:
        async with self.database.session() as session:
            return await BlobRepository(session, self.guid, self.clock).head(id)
