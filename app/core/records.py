from datetime import datetime
from typing import Annotated
from uuid import UUID

from sqlalchemy.orm import Mapped, mapped_column

from lib.database import DatabaseDeclarativeBase


class AppRecord(DatabaseDeclarativeBase):
    __abstract__ = True

    id: Mapped[Annotated[UUID, mapped_column(primary_key=True)]]


class TimestampRecordMixin:
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
