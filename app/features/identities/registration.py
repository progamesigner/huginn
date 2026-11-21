from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from lib.result import Err, Ok, Result

from .entities import IdentityLinkInvite, IssuedIdentityLinkInvite, LinkedPrincipal
from .exceptions import InvalidIdentityLinkError
from .repositories import IdentityRepository
from .sources.sqlalchemy import IdentityLinkInviteRecord
from .tokens import IdentityTokenCodec
from .validation import registration_handle, validate_scope


class RegistrationService:
    def __init__(
        self,
        repository: IdentityRepository,
        tokens: IdentityTokenCodec,
    ) -> None:
        self.repository = repository
        self.tokens = tokens

    async def issue(
        self,
        provider: str,
        realm: str,
        ttl: timedelta,
        created_by: str,
        *,
        new_user: str | None = None,
        existing_scope: str | None = None,
    ) -> IssuedIdentityLinkInvite:
        # Only the operator CLI exposes issuance. A new scope is explicitly approved
        # by its name; an untrusted recipient cannot choose another vault by claiming it.
        if (new_user is None) == (existing_scope is None):
            raise ValueError('choose exactly one of new_user or existing_scope')
        if ttl <= timedelta() or ttl > timedelta(days=1):
            raise ValueError('registration ttl must be between zero and one day')
        handle = registration_handle(new_user) if new_user is not None else None
        scope = handle if handle is not None else validate_scope(existing_scope or '')
        await self.repository.namespace_lock()
        if await self.repository.namespace_taken(scope):
            raise ValueError('memory is already owned; use an account link invite')
        query = select(IdentityLinkInviteRecord.id).where(
            IdentityLinkInviteRecord.memory_scope == scope,
            IdentityLinkInviteRecord.consumed_at.is_(None),
            IdentityLinkInviteRecord.revoked_at.is_(None),
            IdentityLinkInviteRecord.expires_at > self.repository.clock.now(),
        )
        if (await self.repository.session.execute(query)).first() is not None:
            raise ValueError('a live registration already reserves that scope')
        token, code, digest = self.tokens.issue()
        expires = self.repository.clock.now() + ttl
        invite = IdentityLinkInviteRecord(
            id=self.repository.guid.generate(),
            user_id=None,
            provider=provider,
            realm=realm,
            token_digest=digest,
            expires_at=expires,
            created_by=created_by,
            kind='register_new' if handle is not None else 'register_existing',
            memory_scope=scope,
            reserved_handle=handle,
        )
        self.repository.session.add(invite)
        await self.repository.session.flush()
        await self.repository.record_audit(
            'registration_issued',
            provider,
            realm,
            metadata={
                'invite_id': str(invite.id),
                'created_by': created_by,
                'mode': invite.kind,
                'memory_scope': scope,
            },
        )
        return IssuedIdentityLinkInvite(
            invite_id=invite.id,
            token=token,
            short_code=code,
            expires_at=expires,
        )

    async def stage(
        self,
        credential: str,
        provider: str,
        realm: str,
        external: str,
    ) -> Result[bool | LinkedPrincipal, InvalidIdentityLinkError]:
        digest = self.tokens.digest_credential(credential)
        invite = (
            await self.repository.lock_invite(digest) if digest is not None else None
        )
        reason = self.invalid(invite, provider, realm, external)
        if reason is not None:
            return Err(InvalidIdentityLinkError(reason))
        if invite is None:
            raise LookupError('identity transaction lost its record')
        if invite.kind == 'link':
            return Ok(False)
        if invite.consumed_at is not None:
            existing = await self.repository.find_linked(provider, realm, external)
            if existing is not None and existing.user_id == invite.user_id:
                return Ok(existing)
            return Err(InvalidIdentityLinkError('consumed'))
        if await self.repository.find_external(provider, realm, external) is not None:
            return Err(InvalidIdentityLinkError('identity_conflict'))
        await self.repository.stage_registration(invite.id, external)
        await self.repository.record_audit(
            'registration_started',
            provider,
            realm,
            metadata={'invite_id': str(invite.id), 'external_user_id': external},
        )
        return Ok(True)

    def invalid(
        self,
        invite: IdentityLinkInvite | None,
        provider: str,
        realm: str,
        external: str,
    ) -> str | None:
        if invite is None:
            return 'unknown_token'
        if (invite.provider, invite.realm) != (provider, realm):
            return 'realm_mismatch'
        if invite.revoked_at is not None:
            return 'revoked'
        if invite.expires_at <= self.repository.clock.now():
            return 'expired'
        if invite.pending_external_user_id not in (None, external):
            return 'identity_conflict'
        return None

    async def claim(
        self,
        provider: str,
        realm: str,
        external: str,
        handle: str,
        *,
        credential: str | None = None,
        display_name: str | None = None,
    ) -> Result[LinkedPrincipal, InvalidIdentityLinkError]:
        if credential is None:
            invite = await self.repository.pending_registration(
                provider,
                realm,
                external,
            )
        else:
            digest = self.tokens.digest_credential(credential)
            invite = (
                await self.repository.lock_invite(digest)
                if digest is not None
                else None
            )
        reason = self.invalid(invite, provider, realm, external)
        if reason is not None:
            return Err(InvalidIdentityLinkError(reason))
        if invite is None:
            raise LookupError('identity transaction lost its record')
        if invite.kind == 'link':
            return Err(InvalidIdentityLinkError('registration_required'))
        await self.repository.namespace_lock()
        reason = self.invalid(invite, provider, realm, external)
        if reason is not None:
            return Err(InvalidIdentityLinkError(reason))
        existing = await self.repository.find_linked(provider, realm, external)
        if invite.consumed_at is not None:
            if existing is not None and existing.user_id == invite.user_id:
                return Ok(existing)
            return Err(InvalidIdentityLinkError('consumed'))
        if existing is not None:
            return Err(InvalidIdentityLinkError('identity_conflict'))
        try:
            name = registration_handle(handle)
            if invite.reserved_handle is not None and name != invite.reserved_handle:
                return Err(InvalidIdentityLinkError('handle_mismatch'))
            if not display_name or not display_name.strip():
                display_name = name
            if len(display_name) > 128:
                raise ValueError('display name too long')
            async with self.repository.session.begin_nested():
                user = await self.repository.create_user(
                    name,
                    display_name,
                    memory_scope=invite.memory_scope,
                    registration_invite_id=invite.id,
                )
                identity = await self.repository.link(
                    user.id,
                    provider,
                    realm,
                    external,
                )
                record = await self.repository.session.get(
                    IdentityLinkInviteRecord,
                    invite.id,
                )
                if record is None:
                    raise LookupError('identity transaction lost its record')
                record.user_id = user.id
                record.pending_external_user_id = external
                await self.repository.consume_invite(
                    invite.id,
                    self.repository.clock.now(),
                )
                await self.repository.record_audit(
                    'registration_succeeded',
                    provider,
                    realm,
                    user_id=user.id,
                    external_identity_id=identity.id,
                    metadata={'invite_id': str(invite.id)},
                )
        except ValueError, IntegrityError:
            return Err(InvalidIdentityLinkError('name_conflict'))
        principal = await self.repository.find_linked(provider, realm, external)
        if principal is None:
            raise LookupError('identity transaction lost its record')
        return Ok(principal)
