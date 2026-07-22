from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy.exc import IntegrityError

from .exceptions import InvalidIdentityLinkError
from .repositories import IdentityRepository
from .tokens import IdentityTokenCodec
from .types import (
    AnonymousPrincipal,
    ExternalIdentity,
    IdentityLinkInvite,
    IssuedIdentityLinkInvite,
    LinkedPrincipal,
    Principal,
)


class IdentityLinkingService:
    def __init__(
        self,
        repository: IdentityRepository,
        tokens: IdentityTokenCodec,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
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
        now = self.clock()
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
        return IssuedIdentityLinkInvite(token, short_code, invite.expires_at)

    async def complete(
        self,
        credential: str,
        provider: str,
        realm: str,
        external_user_id: str,
        profile: Mapping[str, JsonValue] | None = None,
    ) -> ExternalIdentity:
        digest = self.tokens.digest_credential(credential)
        invite = (
            await self.repository.lock_invite(digest) if digest is not None else None
        )
        if invite is None:
            await self._audit_failure(provider, realm, 'unknown_token')
            raise InvalidIdentityLinkError()

        reason = self._invalid_reason(invite, provider, realm)
        if reason is not None:
            await self._audit_failure(provider, realm, reason)
            raise InvalidIdentityLinkError()

        existing = await self.repository.find_external(
            provider,
            realm,
            external_user_id,
        )
        if invite.consumed_at is not None:
            if existing is not None and existing.user_id == invite.user_id:
                return existing
            await self._audit_failure(provider, realm, 'consumed')
            raise InvalidIdentityLinkError()
        if existing is not None:
            if existing.user_id == invite.user_id:
                await self.repository.consume_invite(invite.id, self.clock())
                return existing
            await self._audit_failure(provider, realm, 'identity_conflict')
            raise InvalidIdentityLinkError()

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
            await self._audit_failure(provider, realm, 'identity_conflict')
            raise InvalidIdentityLinkError() from None

        await self.repository.consume_invite(invite.id, self.clock())
        await self.repository.record_audit(
            'link_succeeded',
            provider,
            realm,
            user_id=invite.user_id,
            external_identity_id=identity.id,
            metadata={'invite_id': str(invite.id)},
        )
        return identity

    async def revoke(self, invite_id: UUID) -> bool:
        invite = await self.repository.lock_invite_by_id(invite_id)
        if (
            invite is None
            or invite.consumed_at is not None
            or invite.revoked_at is not None
        ):
            return False
        await self.repository.revoke_invite(invite.id, self.clock())
        await self.repository.record_audit(
            'link_revoked',
            invite.provider,
            invite.realm,
            user_id=invite.user_id,
            metadata={'invite_id': str(invite.id)},
        )
        return True

    def _invalid_reason(
        self, invite: IdentityLinkInvite, provider: str, realm: str
    ) -> str | None:
        if invite.provider != provider or invite.realm != realm:
            return 'realm_mismatch'
        if invite.revoked_at is not None:
            return 'revoked'
        if invite.expires_at <= self.clock():
            return 'expired'
        return None

    async def _audit_failure(self, provider: str, realm: str, reason: str) -> None:
        await self.repository.record_audit(
            'link_failed',
            provider,
            realm,
            metadata={'reason': reason},
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
        identity = await self.repository.find_external(
            provider,
            realm,
            external_user_id,
        )

        if identity is None:
            return AnonymousPrincipal(provider, realm, external_user_id)

        return LinkedPrincipal(identity.user_id, identity.id)
