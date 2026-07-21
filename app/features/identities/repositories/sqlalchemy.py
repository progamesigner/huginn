from datetime import datetime
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import CheckConstraint, ForeignKey, Index, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.records import AppRecord, TimestampRecordMixin


class UserRecord(TimestampRecordMixin, AppRecord):
    __tablename__ = 'users'
    __table_args__ = (
        CheckConstraint(
            'handle = lower(btrim(handle)) AND length(handle) > 0',
            name='ck_users_handle_normalized',
        ),
    )

    handle: Mapped[str] = mapped_column(unique=True)
    display_name: Mapped[str]


class ExternalIdentityRecord(AppRecord):
    __tablename__ = 'external_identities'
    __table_args__ = (
        UniqueConstraint('provider', 'realm', 'external_user_id'),
        Index('ix_external_identities_user_provider', 'user_id', 'provider'),
    )

    user_id: Mapped[UUID] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'))
    provider: Mapped[str]
    realm: Mapped[str]
    external_user_id: Mapped[str]
    profile: Mapped[JsonValue] = mapped_column()
    linked_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class IdentityAuditEventRecord(AppRecord):
    __tablename__ = 'identity_audit_events'
    __table_args__ = (
        Index('ix_identity_audit_events_user_occurred', 'user_id', 'occurred_at'),
    )

    event_type: Mapped[str]
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL')
    )
    external_identity_id: Mapped[UUID | None]
    provider: Mapped[str]
    realm: Mapped[str]
    occurred_at: Mapped[datetime] = mapped_column(server_default=func.now())
    metadata_: Mapped[JsonValue] = mapped_column('metadata')
