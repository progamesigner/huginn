from collections.abc import Mapping
from datetime import timedelta
from typing import override
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy.exc import IntegrityError

from lib.database import DatabaseContract
from lib.guid import GUIDContract
from lib.result import Err, Ok, Result
from lib.temporal import ClockContract

from .contracts import IdentityLinkingServiceContract, IdentityResolutionServiceContract
from .entities import (
    AnonymousPrincipal,
    ExternalIdentity,
    IdentityLinkInvite,
    IssuedIdentityLinkInvite,
    LinkedPrincipal,
    Principal,
)
from .exceptions import InvalidIdentityLinkError
from .repositories import IdentityRepository
from .tokens import IdentityTokenCodec


class IdentityLinkingService:
    def __init__(
        self,
        repository: IdentityRepository,
        tokens: IdentityTokenCodec,
        clock: ClockContract,
    ) -> None:
        self.repository = repository
        self.tokens = tokens
        self.clock = clock

    async def issue_invite(
        self,
        user_id: UUID,
        provider: str,
        realm: str,
        ttl: timedelta,
        created_by: str,
    ) -> IssuedIdentityLinkInvite:
        if ttl <= timedelta():
            raise ValueError('ttl must be positive')
        now = self.clock.now()
        token, short_code, digest = self.tokens.issue()
        invite = await self.repository.issue_invite(
            user_id,
            provider,
            realm,
            digest,
            now + ttl,
            created_by,
        )
        await self.repository.record_audit(
            'invite_issued',
            provider,
            realm,
            user_id=user_id,
            metadata={'invite_id': str(invite.id), 'created_by': created_by},
        )
        return IssuedIdentityLinkInvite(
            token=token,
            short_code=short_code,
            expires_at=invite.expires_at,
        )

    async def complete(
        self,
        credential: str,
        provider: str,
        realm: str,
        external_user_id: str,
        profile: Mapping[str, JsonValue] | None = None,
    ) -> Result[ExternalIdentity, InvalidIdentityLinkError]:
        digest = self.tokens.digest_credential(credential)
        invite = (
            await self.repository.lock_invite(digest) if digest is not None else None
        )
        if invite is None:
            return Err(InvalidIdentityLinkError('unknown_token'))

        reason = self._invalid_reason(invite, provider, realm)
        if reason is not None:
            return Err(InvalidIdentityLinkError(reason))

        existing = await self.repository.find_external(
            provider,
            realm,
            external_user_id,
        )
        if invite.consumed_at is not None:
            if existing is not None and existing.user_id == invite.user_id:
                return Ok(existing)
            return Err(InvalidIdentityLinkError('consumed'))
        if existing is not None:
            if existing.user_id == invite.user_id:
                await self.repository.consume_invite(invite.id, self.clock.now())
                return Ok(existing)
            return Err(InvalidIdentityLinkError('identity_conflict'))

        try:
            async with self.repository.session.begin_nested():
                identity = await self.repository.link(
                    invite.user_id,
                    provider,
                    realm,
                    external_user_id,
                    profile,
                )
        except IntegrityError:
            return Err(InvalidIdentityLinkError('identity_conflict'))

        await self.repository.consume_invite(invite.id, self.clock.now())
        await self.repository.record_audit(
            'link_succeeded',
            provider,
            realm,
            user_id=invite.user_id,
            external_identity_id=identity.id,
            metadata={'invite_id': str(invite.id)},
        )
        return Ok(identity)

    async def revoke(self, invite_id: UUID) -> bool:
        invite = await self.repository.lock_invite_by_id(invite_id)
        if (
            invite is None
            or invite.consumed_at is not None
            or invite.revoked_at is not None
        ):
            return False
        await self.repository.revoke_invite(invite.id, self.clock.now())
        await self.repository.record_audit(
            'link_revoked',
            invite.provider,
            invite.realm,
            user_id=invite.user_id,
            metadata={'invite_id': str(invite.id)},
        )
        return True

    async def unlink(self, identity: ExternalIdentity) -> bool:
        if not await self.repository.unlink(identity.id):
            return False

        await self.repository.record_audit(
            'link_revoked',
            identity.provider,
            identity.realm,
            user_id=identity.user_id,
            external_identity_id=identity.id,
        )

        return True

    def _invalid_reason(
        self,
        invite: IdentityLinkInvite,
        provider: str,
        realm: str,
    ) -> str | None:
        if invite.provider != provider or invite.realm != realm:
            return 'realm_mismatch'
        if invite.revoked_at is not None:
            return 'revoked'
        if invite.expires_at <= self.clock.now():
            return 'expired'
        return None


class PrincipalLinker(IdentityLinkingServiceContract):
    clock: ClockContract
    database: DatabaseContract
    guid: GUIDContract
    tokens: IdentityTokenCodec

    def __init__(
        self,
        database: DatabaseContract,
        guid: GUIDContract,
        clock: ClockContract,
        tokens: IdentityTokenCodec,
    ) -> None:
        self.clock = clock
        self.database = database
        self.guid = guid
        self.tokens = tokens

    @override
    async def link(
        self,
        credential: str,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> Result[LinkedPrincipal, InvalidIdentityLinkError]:
        async with self.database.transaction() as session:
            repository = IdentityRepository(session, self.guid, self.clock)
            completed = await IdentityLinkingService(
                repository,
                self.tokens,
                self.clock,
            ).complete(credential, provider, realm, external_user_id)
            principal = (
                await repository.find_linked(provider, realm, external_user_id)
                if completed.ok
                else None
            )

        if isinstance(completed, Ok):
            if principal is None:
                raise LookupError('the completed link is not readable back')
            return Ok(principal)

        # NOTE: a rejection writes nothing before it returns, so the audit line goes
        # into a transaction of its own
        await self._audit_failure(
            provider,
            realm,
            external_user_id,
            completed.error.reason,
        )
        return completed

    async def _audit_failure(
        self,
        provider: str,
        realm: str,
        external_user_id: str,
        reason: str,
    ) -> None:
        async with self.database.transaction() as session:
            await IdentityRepository(session, self.guid, self.clock).record_audit(
                'link_failed',
                provider,
                realm,
                metadata={'reason': reason, 'external_user_id': external_user_id},
            )


class PrincipalResolver(IdentityResolutionServiceContract):
    clock: ClockContract
    database: DatabaseContract
    guid: GUIDContract

    def __init__(
        self,
        database: DatabaseContract,
        guid: GUIDContract,
        clock: ClockContract,
    ) -> None:
        self.clock = clock
        self.database = database
        self.guid = guid

    @override
    async def resolve(
        self,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> Principal:
        async with self.database.session() as session:
            repository = IdentityRepository(session, self.guid, self.clock)
            return await IdentityResolver(repository).resolve(
                provider,
                realm,
                external_user_id,
            )


class IdentityResolver:
    def __init__(self, repository: IdentityRepository) -> None:
        self.repository = repository

    async def resolve(
        self,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> Principal:
        linked = await self.repository.find_linked(provider, realm, external_user_id)

        return linked or AnonymousPrincipal(
            provider=provider,
            realm=realm,
            external_user_id=external_user_id,
        )
