from collections.abc import Mapping
from datetime import datetime
from typing import cast
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.identities.types import (
    ExternalIdentity,
    IdentityAuditEvent,
    IdentityLinkInvite,
    User,
)
from lib.guid import GUIDContract
from lib.temporal import ClockContract

from .sqlalchemy import (
    ExternalIdentityRecord,
    IdentityAuditEventRecord,
    IdentityLinkInviteRecord,
    UserRecord,
)


def _to_user(record: UserRecord) -> User:
    return User(
        record.id,
        record.handle,
        record.display_name,
        record.created_at,
        record.updated_at,
    )


def _to_external(record: ExternalIdentityRecord) -> ExternalIdentity:
    return ExternalIdentity(
        record.id,
        record.user_id,
        record.provider,
        record.realm,
        record.external_user_id,
        cast(Mapping[str, JsonValue], record.profile),
        record.linked_at,
        record.updated_at,
    )


def _to_audit(record: IdentityAuditEventRecord) -> IdentityAuditEvent:
    return IdentityAuditEvent(
        id=record.id,
        event_type=record.event_type,
        provider=record.provider,
        realm=record.realm,
        user_id=record.user_id,
        external_identity_id=record.external_identity_id,
        metadata=cast(Mapping[str, JsonValue], record.metadata_),
        occurred_at=record.occurred_at,
    )


def _to_invite(
    record: IdentityLinkInviteRecord,
) -> IdentityLinkInvite:
    return IdentityLinkInvite(
        id=record.id,
        user_id=record.user_id,
        provider=record.provider,
        realm=record.realm,
        expires_at=record.expires_at,
        created_by=record.created_by,
        consumed_at=record.consumed_at,
        revoked_at=record.revoked_at,
        created_at=record.created_at,
    )


class IdentityRepository:
    def __init__(
        self,
        session: AsyncSession,
        guid: GUIDContract,
        clock: ClockContract,
    ) -> None:
        self.session = session
        self.guid = guid
        self.clock = clock

    async def create_user(self, handle: str, display_name: str) -> User:
        normalized = handle.strip().lower()
        if not normalized:
            raise ValueError('handle must not be empty')
        now = self.clock.now()
        record = UserRecord(
            id=self.guid.generate(),
            handle=normalized,
            display_name=display_name,
            created_at=now,
            updated_at=now,
        )
        self.session.add(record)
        await self.session.flush()
        return _to_user(record)

    async def find_user(self, handle: str) -> User | None:
        query = select(UserRecord).where(UserRecord.handle == handle.strip().lower())
        record = (await self.session.execute(query)).scalar_one_or_none()
        return _to_user(record) if record else None

    async def find_external(
        self, provider: str, realm: str, external_user_id: str
    ) -> ExternalIdentity | None:
        query = select(ExternalIdentityRecord).where(
            ExternalIdentityRecord.provider == provider,
            ExternalIdentityRecord.realm == realm,
            ExternalIdentityRecord.external_user_id == external_user_id,
        )
        record = (await self.session.execute(query)).scalar_one_or_none()
        return _to_external(record) if record else None

    async def list_external(self, user_id: UUID) -> tuple[ExternalIdentity, ...]:
        query = (
            select(ExternalIdentityRecord)
            .where(ExternalIdentityRecord.user_id == user_id)
            .order_by(
                ExternalIdentityRecord.provider,
                ExternalIdentityRecord.realm,
                ExternalIdentityRecord.external_user_id,
            )
        )
        records = (await self.session.execute(query)).scalars().all()
        return tuple(_to_external(record) for record in records)

    async def link(
        self,
        user_id: UUID,
        provider: str,
        realm: str,
        external_user_id: str,
        profile: Mapping[str, JsonValue] | None = None,
    ) -> ExternalIdentity:
        record = ExternalIdentityRecord(
            id=self.guid.generate(),
            user_id=user_id,
            provider=provider,
            realm=realm,
            external_user_id=external_user_id,
            profile=dict(profile or {}),
        )
        self.session.add(record)
        await self.session.flush()
        return _to_external(record)

    async def unlink(self, identity_id: UUID) -> bool:
        record = await self.session.get(ExternalIdentityRecord, identity_id)
        if record is None:
            return False
        await self.session.delete(record)
        await self.session.flush()
        return True

    async def record_audit(
        self,
        event_type: str,
        provider: str,
        realm: str,
        *,
        user_id: UUID | None = None,
        external_identity_id: UUID | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> IdentityAuditEvent:
        record = IdentityAuditEventRecord(
            id=self.guid.generate(),
            event_type=event_type,
            provider=provider,
            realm=realm,
            user_id=user_id,
            external_identity_id=external_identity_id,
            metadata_=dict(metadata or {}),
        )
        self.session.add(record)
        await self.session.flush()
        return _to_audit(record)

    async def issue_invite(
        self,
        user_id: UUID,
        provider: str,
        realm: str,
        token_digest: bytes,
        expires_at: datetime,
        created_by: str,
    ) -> IdentityLinkInvite:
        record = IdentityLinkInviteRecord(
            id=self.guid.generate(),
            user_id=user_id,
            provider=provider,
            realm=realm,
            token_digest=token_digest,
            expires_at=expires_at,
            created_by=created_by,
        )
        self.session.add(record)
        await self.session.flush()
        return _to_invite(record)

    async def lock_invite(self, token_digest: bytes) -> IdentityLinkInvite | None:
        query = (
            select(IdentityLinkInviteRecord)
            .where(IdentityLinkInviteRecord.token_digest == token_digest)
            .with_for_update()
        )
        record = (await self.session.execute(query)).scalar_one_or_none()
        return _to_invite(record) if record else None

    async def lock_invite_by_id(self, invite_id: UUID) -> IdentityLinkInvite | None:
        query = (
            select(IdentityLinkInviteRecord)
            .where(IdentityLinkInviteRecord.id == invite_id)
            .with_for_update()
        )
        record = (await self.session.execute(query)).scalar_one_or_none()
        return _to_invite(record) if record else None

    async def consume_invite(self, invite_id: UUID, consumed_at: datetime) -> None:
        record = await self.session.get(IdentityLinkInviteRecord, invite_id)
        if record is None:
            raise LookupError('identity link invite not found')
        record.consumed_at = consumed_at
        await self.session.flush()

    async def revoke_invite(self, invite_id: UUID, revoked_at: datetime) -> None:
        record = await self.session.get(IdentityLinkInviteRecord, invite_id)
        if record is None:
            raise LookupError('identity link invite not found')
        record.revoked_at = revoked_at
        await self.session.flush()
