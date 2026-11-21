from datetime import datetime
from typing import Annotated
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    UniqueConstraint,
    column,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.records import AppRecord
from lib.database import Timestamp

_KINDS = ('audio', 'document', 'image', 'video')


class BlobRecord(AppRecord):
    __tablename__ = 'blobs'
    __table_args__ = (
        UniqueConstraint('sha256'),
        CheckConstraint(column('kind').in_(_KINDS), name='kind_known'),
        Index('ix_blobs_expires_at', 'expires_at'),
    )

    sha256: Mapped[str]
    mime: Mapped[str]
    size: Mapped[int]
    data: Mapped[bytes]
    kind: Mapped[str]
    filename: Mapped[Annotated[str | None, mapped_column(default=None)]]
    expires_at: Mapped[
        Annotated[datetime | None, mapped_column(Timestamp(), default=None)]
    ]
    created_at: Mapped[Annotated[datetime, mapped_column(server_default=func.now())]]


# NOTE: blobs are shared by content, so who may fetch one cannot live on the row itself
class BlobGrantRecord(AppRecord):
    __tablename__ = 'blob_grants'
    __table_args__ = (UniqueConstraint('blob_id', 'owner'),)

    blob_id: Mapped[
        Annotated[
            UUID,
            mapped_column(
                ForeignKey(BlobRecord.id, ondelete='CASCADE'),
            ),
        ]
    ]
    owner: Mapped[str]
    filename: Mapped[Annotated[str | None, mapped_column(default=None)]]
    created_at: Mapped[Annotated[datetime, mapped_column(server_default=func.now())]]
