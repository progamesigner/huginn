from datetime import datetime

from sqlalchemy import CheckConstraint, Index, UniqueConstraint, column, func
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

    sha256: Mapped[str] = mapped_column()
    mime: Mapped[str] = mapped_column()
    size: Mapped[int] = mapped_column()
    data: Mapped[bytes] = mapped_column()
    kind: Mapped[str] = mapped_column()
    filename: Mapped[str | None] = mapped_column(default=None)
    expires_at: Mapped[datetime | None] = mapped_column(Timestamp(), default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
