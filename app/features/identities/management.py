from datetime import timedelta
from typing import override

from lib.database import DatabaseContract
from lib.guid import GUIDContract
from lib.result import Ok, Result
from lib.temporal import ClockContract

from .contracts import IdentityManagementServiceContract, IdentityTargetServiceContract
from .entities import IssuedIdentityLinkInvite, LinkedPrincipal
from .exceptions import InvalidIdentityLinkError
from .registration import RegistrationService
from .repositories import IdentityRepository
from .services import IdentityLinkingService
from .tokens import IdentityTokenCodec


class IdentityManagementService(IdentityManagementServiceContract):
    def __init__(
        self,
        database: DatabaseContract,
        guid: GUIDContract,
        clock: ClockContract,
        tokens: IdentityTokenCodec,
        targets: IdentityTargetServiceContract,
    ) -> None:
        self.database = database
        self.guid = guid
        self.clock = clock
        self.tokens = tokens
        self.targets = targets

    @override
    async def register(
        self,
        credential: str,
        provider: str,
        realm: str,
        external: str,
    ) -> Result[LinkedPrincipal | None, InvalidIdentityLinkError]:
        async with self.database.transaction() as session:
            repo = IdentityRepository(session, self.guid, self.clock)
            staged = await RegistrationService(repo, self.tokens).stage(
                credential,
                provider,
                realm,
                external,
            )
            if not isinstance(staged, Ok):
                await self._failure(
                    repo,
                    'registration_failed',
                    provider,
                    realm,
                    external,
                    staged.error.reason,
                )
                return staged
            if isinstance(staged.value, LinkedPrincipal):
                return Ok(staged.value)
            if staged.value:
                return Ok(None)
            linked = await IdentityLinkingService(
                repo,
                self.tokens,
                self.clock,
            ).complete(
                credential,
                provider,
                realm,
                external,
            )
            if not isinstance(linked, Ok):
                await self._failure(
                    repo,
                    'link_failed',
                    provider,
                    realm,
                    external,
                    linked.error.reason,
                )
                return linked
            principal = await repo.find_linked(provider, realm, external)
            if principal is None:
                raise LookupError('identity transaction lost its record')
            return Ok(principal)

    @override
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
        async with self.database.transaction() as session:
            repo = IdentityRepository(session, self.guid, self.clock)
            claimed = await RegistrationService(repo, self.tokens).claim(
                provider,
                realm,
                external,
                handle,
                credential=credential,
                display_name=display_name,
            )
            if not isinstance(claimed, Ok):
                await self._failure(
                    repo,
                    'registration_failed',
                    provider,
                    realm,
                    external,
                    claimed.error.reason,
                )
            return claimed

    @override
    async def invite(
        self,
        provider: str,
        realm: str,
        external: str,
        target: str,
    ) -> IssuedIdentityLinkInvite:
        async with self.database.transaction() as session:
            repo = IdentityRepository(session, self.guid, self.clock)
            try:
                principal = await repo.find_linked(provider, realm, external)
                if principal is None:
                    raise ValueError(
                        'link your account before inviting another channel',
                    )
                target_provider, target_realm = self.targets.resolve(target)
                return await IdentityLinkingService(
                    repo,
                    self.tokens,
                    self.clock,
                ).issue_invite(
                    principal.user_id,
                    target_provider,
                    target_realm,
                    timedelta(minutes=10),
                    principal.stable_key,
                )
            except ValueError:
                await self._failure(
                    repo,
                    'invite_failed',
                    provider,
                    realm,
                    external,
                    'invalid_target_or_identity',
                )
        raise ValueError('link your account and choose a configured target')

    @override
    async def display_name(
        self,
        provider: str,
        realm: str,
        external: str,
        name: str,
    ) -> None:
        async with self.database.transaction() as session:
            repo = IdentityRepository(session, self.guid, self.clock)
            principal = await repo.find_linked(provider, realm, external)
            try:
                if principal is None:
                    raise ValueError('link your account first')
                await repo.update_display_name(principal.user_id, name)
            except ValueError:
                await self._failure(
                    repo,
                    'display_name_failed',
                    provider,
                    realm,
                    external,
                    'invalid_name_or_identity',
                )
            else:
                await repo.record_audit(
                    'display_name_updated',
                    provider,
                    realm,
                    user_id=principal.user_id,
                )
                return
        raise ValueError(
            'link your account and use a display name of 1 to 128 characters',
        )

    async def _failure(
        self,
        repo: IdentityRepository,
        event: str,
        provider: str,
        realm: str,
        external: str,
        reason: str,
    ) -> None:
        await repo.record_audit(
            event,
            provider,
            realm,
            metadata={
                'external_user_id': external,
                'reason': reason,
            },
        )
