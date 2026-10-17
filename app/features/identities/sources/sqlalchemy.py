from datetime import datetime
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.records import AppRecord, TimestampRecordMixin
from lib.database import Timestamp

# NOTE: pydantic's JsonValue admits None, so SQLAlchemy infers a nullable column from
# the annotation alone; every JSON column here is NOT NULL and has to say so
_EMPTY_JSON = text("'{}'::jsonb")


class UserRecord(TimestampRecordMixin, AppRecord):
    __tablename__ = 'users'
    __table_args__ = (
        CheckConstraint(
            'handle = lower(btrim(handle)) AND handle ~ '
            "'^[a-z0-9._-]{1,64}$' AND handle NOT IN ('.', '..')",
            name='handle_normalized',
        ),
    )

    handle: Mapped[str] = mapped_column(unique=True)
    display_name: Mapped[str] = mapped_column()
    memory_scope: Mapped[str] = mapped_column(unique=True)


class ExternalIdentityRecord(AppRecord):
    __tablename__ = 'external_identities'
    __table_args__ = (
        UniqueConstraint('provider', 'realm', 'external_user_id'),
        Index('ix_external_identities_user_provider', 'user_id', 'provider'),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey(UserRecord.id, ondelete='CASCADE'),
    )
    provider: Mapped[str] = mapped_column()
    realm: Mapped[str] = mapped_column()
    external_user_id: Mapped[str] = mapped_column()
    profile: Mapped[JsonValue] = mapped_column(
        nullable=False,
        server_default=_EMPTY_JSON,
    )
    linked_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now(),
    )


class IdentityAuditEventRecord(AppRecord):
    __tablename__ = 'identity_audit_events'
    __table_args__ = (
        Index('ix_identity_audit_events_user_occurred', 'user_id', 'occurred_at'),
    )

    event_type: Mapped[str] = mapped_column()
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(UserRecord.id, ondelete='SET NULL'),
    )
    external_identity_id: Mapped[UUID | None] = mapped_column()
    provider: Mapped[str] = mapped_column()
    realm: Mapped[str] = mapped_column()
    occurred_at: Mapped[datetime] = mapped_column(server_default=func.now())
    metadata_: Mapped[JsonValue] = mapped_column(
        'metadata',
        nullable=False,
        server_default=_EMPTY_JSON,
    )


class IdentityLinkInviteRecord(AppRecord):
    __tablename__ = 'identity_link_invites'
    __table_args__ = (
        UniqueConstraint('token_digest'),
        CheckConstraint(
            'NOT (consumed_at IS NOT NULL AND revoked_at IS NOT NULL)',
            name='terminal_state',
        ),
        Index('ix_identity_link_invites_user_created', 'user_id', 'created_at'),
    )

    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(UserRecord.id, ondelete='CASCADE'),
    )
    provider: Mapped[str] = mapped_column()
    realm: Mapped[str] = mapped_column()
    token_digest: Mapped[bytes] = mapped_column()
    expires_at: Mapped[datetime] = mapped_column(Timestamp())
    consumed_at: Mapped[datetime | None] = mapped_column(Timestamp())
    revoked_at: Mapped[datetime | None] = mapped_column(Timestamp())
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    created_by: Mapped[str] = mapped_column()
    kind: Mapped[str] = mapped_column(server_default='link')
    memory_scope: Mapped[str | None] = mapped_column()
    reserved_handle: Mapped[str | None] = mapped_column()
    pending_external_user_id: Mapped[str | None] = mapped_column()
