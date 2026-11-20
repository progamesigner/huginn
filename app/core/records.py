from datetime import datetime
from typing import ClassVar
from uuid import UUID

from sqlalchemy import Table
from sqlalchemy.orm import Mapped, mapped_column

from lib.database import DatabaseDeclarativeBase


class AppRecord(DatabaseDeclarativeBase):
    # SQLAlchemy subclasses expose Table, while the base accepts any FromClause.
    __table__: ClassVar[Table]  # ty: ignore[incompatible-mutable-override]

    __abstract__ = True

    id: Mapped[UUID] = mapped_column(primary_key=True)


class TimestampRecordMixin:
    created_at: Mapped[datetime] = mapped_column()
    updated_at: Mapped[datetime] = mapped_column()
