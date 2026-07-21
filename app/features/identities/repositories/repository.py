from collections.abc import Mapping
from typing import cast
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.identities.types import ExternalIdentity, IdentityAuditEvent, User
from lib.guid import GUIDContract

from .sqlalchemy import ExternalIdentityRecord, IdentityAuditEventRecord, UserRecord


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


class IdentityRepository:
    def __init__(self, session: AsyncSession, guid: GUIDContract) -> None:
        self.session = session
        self.guid = guid

    async def create_user(self, handle: str, display_name: str) -> User:
        normalized = handle.strip().lower()
        if not normalized:
            raise ValueError('handle must not be empty')
        record = UserRecord(
            id=self.guid.generate(), handle=normalized, display_name=display_name
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
