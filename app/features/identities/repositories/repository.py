from collections.abc import Mapping
from datetime import datetime
from typing import cast
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.identities.entities import (
    ExternalIdentity,
    IdentityAuditEvent,
    IdentityLinkInvite,
    LinkedPrincipal,
    User,
)
from app.features.identities.sources.sqlalchemy import (
    ExternalIdentityRecord,
    IdentityAuditEventRecord,
    IdentityLinkInviteRecord,
    UserRecord,
)
from app.features.identities.validation import normalize_handle, validate_scope
from lib.guid import GUIDContract
from lib.temporal import ClockContract


def _to_user(record: UserRecord) -> User:
    return User(
        id=record.id,
        handle=record.handle,
        display_name=record.display_name,
        memory_scope=record.memory_scope,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_external(record: ExternalIdentityRecord) -> ExternalIdentity:
    return ExternalIdentity(
        id=record.id,
        user_id=record.user_id,
        provider=record.provider,
        realm=record.realm,
        external_user_id=record.external_user_id,
        profile=cast(Mapping[str, JsonValue], record.profile),
        linked_at=record.linked_at,
        updated_at=record.updated_at,
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
        kind=record.kind,
        memory_scope=record.memory_scope,
        reserved_handle=record.reserved_handle,
        pending_external_user_id=record.pending_external_user_id,
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

    async def namespace_lock(self) -> None:
        await self.session.execute(text('SELECT pg_advisory_xact_lock(726451201)'))

    async def namespace_taken(self, value: str) -> bool:
        query = select(UserRecord.id).where(
            or_(UserRecord.handle == value, UserRecord.memory_scope == value),
        )
        return (await self.session.execute(query)).first() is not None

    async def create_user(
        self,
        handle: str,
        display_name: str,
        *,
        memory_scope: str | None = None,
        registration_invite_id: UUID | None = None,
    ) -> User:
        normalized = normalize_handle(handle)
        scope = validate_scope(memory_scope or normalized)
        await self.namespace_lock()
        if await self.namespace_taken(scope) or await self.namespace_taken(normalized):
            raise ValueError('handle or memory scope is already owned')
        pending = select(IdentityLinkInviteRecord.id).where(
            IdentityLinkInviteRecord.memory_scope.in_({scope, normalized}),
            IdentityLinkInviteRecord.consumed_at.is_(None),
            IdentityLinkInviteRecord.revoked_at.is_(None),
            IdentityLinkInviteRecord.expires_at > self.clock.now(),
        )
        if registration_invite_id is not None:
            pending = pending.where(
                IdentityLinkInviteRecord.id != registration_invite_id,
            )
        if (await self.session.execute(pending)).first() is not None:
            raise ValueError('handle or memory scope is reserved by an invitation')
        now = self.clock.now()
        record = UserRecord(
            id=self.guid.generate(),
            handle=normalized,
            memory_scope=scope,
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

    async def find_linked(
        self,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> LinkedPrincipal | None:
        query = (
            select(
                ExternalIdentityRecord.user_id,
                ExternalIdentityRecord.id,
                UserRecord.handle,
                UserRecord.memory_scope,
                UserRecord.display_name,
            )
            .join(UserRecord, UserRecord.id == ExternalIdentityRecord.user_id)
            .where(
                ExternalIdentityRecord.provider == provider,
                ExternalIdentityRecord.realm == realm,
                ExternalIdentityRecord.external_user_id == external_user_id,
            )
        )
        row = (await self.session.execute(query)).one_or_none()
        return (
            None
            if row is None
            else LinkedPrincipal(
                user_id=row.user_id,
                external_identity_id=row.id,
                handle=row.handle,
                memory_scope=row.memory_scope,
                display_name=row.display_name,
            )
        )

    async def find_external(
        self,
        provider: str,
        realm: str,
        external_user_id: str,
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

    async def consume_invite(
        self,
        invite_id: UUID,
        consumed_at: datetime,
    ) -> IdentityLinkInvite | None:
        record = await self.session.get(IdentityLinkInviteRecord, invite_id)
        if record is None:
            return None
        record.consumed_at = consumed_at
        await self.session.flush()
        return _to_invite(record)

    async def revoke_invite(
        self,
        invite_id: UUID,
        revoked_at: datetime,
    ) -> IdentityLinkInvite | None:
        record = await self.session.get(IdentityLinkInviteRecord, invite_id)
        if record is None:
            return None
        record.revoked_at = revoked_at
        await self.session.flush()
        return _to_invite(record)

    async def stage_registration(self, invite_id: UUID, external: str) -> None:
        record = await self.session.get(IdentityLinkInviteRecord, invite_id)
        if record is None:
            raise LookupError('identity transaction lost its record')
        record.pending_external_user_id = external
        await self.session.flush()

    async def pending_registration(
        self,
        provider: str,
        realm: str,
        external: str,
    ) -> IdentityLinkInvite | None:
        query = (
            select(IdentityLinkInviteRecord)
            .where(
                IdentityLinkInviteRecord.provider == provider,
                IdentityLinkInviteRecord.realm == realm,
                IdentityLinkInviteRecord.pending_external_user_id == external,
                IdentityLinkInviteRecord.user_id.is_(None),
                IdentityLinkInviteRecord.revoked_at.is_(None),
                IdentityLinkInviteRecord.expires_at > self.clock.now(),
            )
            .order_by(IdentityLinkInviteRecord.created_at.desc())
            .with_for_update()
        )
        record = (await self.session.execute(query)).scalars().first()
        return _to_invite(record) if record else None

    async def update_display_name(self, user_id: UUID, name: str) -> None:
        if not name.strip() or len(name) > 128:
            raise ValueError('display name must contain 1 to 128 characters')
        record = await self.session.get(UserRecord, user_id)
        if record is None:
            raise LookupError('user not found')
        record.display_name = name.strip()
        record.updated_at = self.clock.now()
        await self.session.flush()
