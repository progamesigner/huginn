from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Mapped, mapped_column

from lib.database import DatabaseDeclarativeBase


class AppRecord(DatabaseDeclarativeBase):
    __abstract__ = True

    id: Mapped[UUID] = mapped_column(primary_key=True)


class TimestampRecordMixin:
    created_at: Mapped[datetime] = mapped_column()
    updated_at: Mapped[datetime] = mapped_column()
