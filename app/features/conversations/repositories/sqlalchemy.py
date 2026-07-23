from datetime import datetime
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.records import AppRecord, TimestampRecordMixin


class ConversationRecord(TimestampRecordMixin, AppRecord):
    __tablename__ = 'conversations'
    __table_args__ = (UniqueConstraint('channel', 'chat_id', 'agent'),)

    channel: Mapped[str]
    chat_id: Mapped[str]
    agent: Mapped[str]
    compact_version: Mapped[int] = mapped_column(server_default='0')
    summarized_through_turn_id: Mapped[UUID | None] = mapped_column(
        ForeignKey('turns.id', ondelete='SET NULL'),
        default=None,
    )


class TurnRecord(AppRecord):
    __tablename__ = 'turns'

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

    turn_id: Mapped[UUID] = mapped_column(ForeignKey('turns.id', ondelete='CASCADE'))
    seq: Mapped[int]
    role: Mapped[str]
    kind: Mapped[str]
    payload: Mapped[JsonValue] = mapped_column()
