from hashlib import sha256
from typing import override

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.conversations.contracts import BlobStoreContract
from app.features.conversations.schemas.sqlalchemy import BlobRecord
from app.features.conversations.types import Blob
from lib.guid import GUIDContract


def _to_blob(record: BlobRecord) -> Blob:
    return Blob(
        id=record.id,
        sha256=record.sha256,
        mime=record.mime,
        size=record.size,
    )


class BlobRepository(BlobStoreContract):
    guid: GUIDContract
    session: AsyncSession

    def __init__(self, session: AsyncSession, guid: GUIDContract) -> None:
        self.guid = guid
        self.session = session

    @override
    async def store(self, mime: str, data: bytes) -> Blob:
        digest = sha256(data).hexdigest()
        statement = (
            insert(BlobRecord)
            .values(
                id=self.guid.generate(),
                sha256=digest,
                mime=mime,
                size=len(data),
                data=data,
            )
            .on_conflict_do_nothing(index_elements=[BlobRecord.sha256])
            .returning(BlobRecord)
        )
        record = (await self.session.execute(statement)).scalar_one_or_none()
        if record is None:
            query = select(BlobRecord).where(BlobRecord.sha256 == digest)
            record = (await self.session.execute(query)).scalar_one()

        return _to_blob(record)
