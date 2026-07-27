from datetime import datetime
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import (
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.records import AppRecord, TimestampRecordMixin

# NOTE: pydantic's JsonValue admits None, so SQLAlchemy infers a nullable column from
# the annotation alone; every JSON column here is NOT NULL and has to say so
_EMPTY_JSON = text("'{}'::jsonb")


class ConversationRecord(TimestampRecordMixin, AppRecord):
    __tablename__ = 'conversations'
    __table_args__ = (UniqueConstraint('channel', 'chat_id', 'agent'),)

    channel: Mapped[str] = mapped_column()
    chat_id: Mapped[str] = mapped_column()
    agent: Mapped[str] = mapped_column()
    compact_version: Mapped[int] = mapped_column(server_default='0')
    summarized_through_turn_id: Mapped[UUID | None] = mapped_column(default=None)


class TurnRecord(AppRecord):
    __tablename__ = 'turns'
    __table_args__ = (Index('ix_turns_conversation_id', 'conversation_id', 'id'),)

    conversation_id: Mapped[UUID] = mapped_column()
    usage: Mapped[JsonValue] = mapped_column(
        nullable=False,
        server_default=_EMPTY_JSON,
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    parts: Mapped[list[PartRecord]] = relationship(
        order_by=lambda: PartRecord.seq,
        cascade='all, delete-orphan',
    )


class PartRecord(AppRecord):
    __tablename__ = 'parts'
    __table_args__ = (Index('ix_parts_turn_id', 'turn_id', 'seq'),)

    turn_id: Mapped[UUID] = mapped_column(
        ForeignKey(TurnRecord.id, ondelete='CASCADE'),
    )
    seq: Mapped[int] = mapped_column()
    role: Mapped[str] = mapped_column()
    kind: Mapped[str] = mapped_column()
    payload: Mapped[JsonValue] = mapped_column(nullable=False)


ConversationRecord.__table__.append_constraint(
    ForeignKeyConstraint(
        [ConversationRecord.summarized_through_turn_id],
        [TurnRecord.id],
        ondelete='SET NULL',
    ),
)

TurnRecord.__table__.append_constraint(
    ForeignKeyConstraint(
        [TurnRecord.conversation_id],
        [ConversationRecord.id],
        ondelete='CASCADE',
    ),
)
