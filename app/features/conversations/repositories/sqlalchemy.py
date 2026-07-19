from datetime import datetime
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.records import AppRecord


class ConversationRecord(AppRecord):
    __tablename__ = 'conversations'
    __table_args__ = (UniqueConstraint('channel', 'chat_id', 'agent'),)

    id: Mapped[UUID] = mapped_column(primary_key=True)
    channel: Mapped[str]
    chat_id: Mapped[str]
    agent: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now(),
    )


class TurnRecord(AppRecord):
    __tablename__ = 'turns'

    id: Mapped[UUID] = mapped_column(primary_key=True)
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey('conversations.id', ondelete='CASCADE'),
    )
    usage: Mapped[JsonValue] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    parts: Mapped[list[PartRecord]] = relationship(
        order_by='PartRecord.seq',
        cascade='all, delete-orphan',
    )


class PartRecord(AppRecord):
    __tablename__ = 'parts'

    id: Mapped[UUID] = mapped_column(primary_key=True)
    turn_id: Mapped[UUID] = mapped_column(ForeignKey('turns.id', ondelete='CASCADE'))
    seq: Mapped[int]
    role: Mapped[str]
    kind: Mapped[str]
    payload: Mapped[JsonValue] = mapped_column()
