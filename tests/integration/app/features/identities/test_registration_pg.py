import asyncio
from datetime import timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.identities.contracts import IdentityTargetServiceContract
from app.features.identities.entities import LinkedPrincipal
from app.features.identities.management import IdentityManagementService
from app.features.identities.registration import RegistrationService
from app.features.identities.repositories import IdentityRepository
from app.features.identities.services import IdentityLinkingService, PrincipalLinker
from app.features.identities.sources.sqlalchemy import (
    ExternalIdentityRecord,
    IdentityAuditEventRecord,
    IdentityLinkInviteRecord,
    UserRecord,
)
from app.features.identities.tokens import IdentityTokenCodec
from lib.database import DatabaseManager
from lib.guid import GUIDManager
from lib.result import Err, Ok
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration
_TOKENS = IdentityTokenCodec(b'r' * 32)


def repo(session: AsyncSession) -> IdentityRepository:
    return IdentityRepository(session, GUIDManager(), ClockManager())


def name() -> str:
    return 'person-' + uuid4().hex


async def issued(session: AsyncSession, *, existing: bool = False):
    scope = name()
    issued = await RegistrationService(repo(session), _TOKENS).issue(
        'http',
        'portal',
        timedelta(minutes=10),
        'test-operator',
        **({'existing_scope': scope} if existing else {'new_user': scope}),
    )
    return scope, issued


@pytest.mark.parametrize('existing', [False, True])
async def test_claim_creates_one_account_identity_and_approved_memory_atomically(
    session: AsyncSession,
    existing: bool,
) -> None:
    scope, invite = await issued(session, existing=existing)
    handle = name() if existing else scope
    service = RegistrationService(repo(session), _TOKENS)
    result = await service.claim(
        'http',
        'portal',
        'recipient',
        handle,
        credential=invite.token,
    )
    assert isinstance(result, Ok)
    principal = result.unwrap()
    assert (principal.handle, principal.memory_scope) == (handle, scope)
    assert principal.stable_key == f'user:{principal.user_id}'
    assert await repo(session).find_linked('http', 'portal', 'recipient') == principal
    digest = _TOKENS.digest_credential(invite.token)
    assert digest is not None
    stored = await repo(session).lock_invite(digest)
    assert stored is not None
    assert stored.consumed_at is not None
    assert stored.user_id == principal.user_id
    assert isinstance(
        await service.claim(
            'http',
            'portal',
            'recipient',
            handle,
            credential=invite.short_code,
        ),
        Ok,
    )
    assert isinstance(
        await service.claim(
            'http',
            'portal',
            'someone-else',
            handle,
            credential=invite.token,
        ),
        Err,
    )


async def test_staged_telegram_registration_requires_invite_and_cannot_be_stolen(
    session: AsyncSession,
) -> None:
    handle, invite = await issued(session)
    service = RegistrationService(repo(session), _TOKENS)
    assert isinstance(await service.claim('http', 'portal', 'recipient', handle), Err)
    assert isinstance(
        await service.stage(invite.short_code, 'http', 'portal', 'recipient'),
        Ok,
    )
    assert isinstance(await service.stage(invite.token, 'http', 'portal', 'thief'), Err)
    assert isinstance(
        await service.claim('http', 'portal', 'recipient', 'other-name'),
        Err,
    )
    result = await service.claim('http', 'portal', 'recipient', handle)
    assert isinstance(result, Ok)
    assert result.unwrap().memory_scope == handle


@pytest.mark.parametrize(
    'problem',
    [
        'wrong_provider',
        'wrong_realm',
        'expired',
        'revoked',
        'taken_handle',
        'taken_identity',
    ],
)
async def test_rejected_registration_does_not_create_an_account_or_consume_invite(
    session: AsyncSession,
    problem: str,
) -> None:
    scope, invite = await issued(session, existing=True)
    repository = repo(session)
    handle = name()
    provider, realm = 'http', 'portal'
    digest = _TOKENS.digest_credential(invite.token)
    assert digest is not None
    stored = await repository.lock_invite(digest)
    assert stored is not None
    record = await session.get(IdentityLinkInviteRecord, stored.id)
    assert record is not None
    if problem == 'wrong_provider':
        provider = 'telegram'
    elif problem == 'wrong_realm':
        realm = 'other-app'
    elif problem == 'expired':
        record.expires_at = ClockManager().now() - timedelta(seconds=1)
    elif problem == 'revoked':
        await repository.revoke_invite(stored.id, ClockManager().now())
    elif problem in {'taken_handle', 'taken_identity'}:
        user = await repository.create_user(handle, 'Fictional person')
        if problem == 'taken_identity':
            await repository.link(user.id, provider, realm, 'recipient')
            handle = name()
    await session.flush()
    before = await session.scalar(select(func.count()).select_from(UserRecord))
    result = await RegistrationService(repository, _TOKENS).claim(
        provider,
        realm,
        'recipient',
        handle,
        credential=invite.token,
    )
    assert isinstance(result, Err)
    assert await session.scalar(select(func.count()).select_from(UserRecord)) == before
    unconsumed = await repository.lock_invite(digest)
    assert unconsumed is not None
    assert unconsumed.consumed_at is None
    assert not await repository.namespace_taken(scope)


async def test_only_operators_approve_unowned_memory_and_owned_memory_uses_link(
    session: AsyncSession,
) -> None:
    repository = repo(session)
    owner = await repository.create_user(name(), 'Fictional Owner')
    registration = RegistrationService(repository, _TOKENS)
    with pytest.raises(ValueError, match='already owned'):
        await registration.issue(
            'http',
            'portal',
            timedelta(minutes=10),
            'operator',
            existing_scope=owner.memory_scope,
        )
    _, invite = await issued(session)
    digest = _TOKENS.digest_credential(invite.token)
    assert digest is not None
    stored = await repository.lock_invite(digest)
    assert stored is not None
    with pytest.raises(ValueError, match='reserves'):
        await registration.issue(
            'http',
            'portal',
            timedelta(minutes=10),
            'operator',
            existing_scope=stored.memory_scope,
        )
    refused = await IdentityLinkingService(
        repository,
        _TOKENS,
        ClockManager(),
    ).complete(invite.token, 'http', 'portal', 'recipient')
    assert not isinstance(refused, Ok)
    assert refused.error.reason == 'link_required'


async def test_display_name_preserves_handle_uuid_and_memory(
    session: AsyncSession,
) -> None:
    scope, invite = await issued(session, existing=True)
    principal = (
        await RegistrationService(repo(session), _TOKENS).claim(
            'http',
            'portal',
            'recipient',
            name(),
            credential=invite.token,
        )
    ).unwrap()
    await repo(session).update_display_name(principal.user_id, 'A New Display Name')
    updated = await repo(session).find_linked('http', 'portal', 'recipient')
    assert updated is not None
    assert updated.display_name == 'A New Display Name'
    assert (updated.handle, updated.user_id, updated.memory_scope) == (
        principal.handle,
        principal.user_id,
        scope,
    )


@pytest.mark.usefixtures('_migrated')
async def test_concurrent_redemption_commits_one_owner_and_audits_both_results(
    database_url: str,
) -> None:
    database = DatabaseManager({None: database_url})
    targets = MagicMock(spec=IdentityTargetServiceContract)
    targets.resolve.return_value = ('telegram', 'next-bot')
    management = IdentityManagementService(
        database,
        GUIDManager(),
        ClockManager(),
        _TOKENS,
        targets,
    )
    try:
        async with database.transaction() as transaction:
            handle, invite = await issued(transaction)
        results = await asyncio.gather(
            *(
                management.claim(
                    'http',
                    'portal',
                    external,
                    handle,
                    credential=invite.token,
                )
                for external in ('first-' + handle, 'second-' + handle)
            ),
        )
        assert sum(isinstance(result, Ok) for result in results) == 1
        winner = next(result.unwrap() for result in results if isinstance(result, Ok))
        assert isinstance(winner, LinkedPrincipal)
        async with database.session() as session:
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(UserRecord)
                    .where(UserRecord.handle == handle),
                )
                == 1
            )
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(ExternalIdentityRecord)
                    .where(ExternalIdentityRecord.user_id == winner.user_id),
                )
                == 1
            )
            audit = (
                await session.scalars(
                    select(IdentityAuditEventRecord).where(
                        IdentityAuditEventRecord.event_type == 'registration_failed',
                    ),
                )
            ).all()
            assert any(
                isinstance(event.metadata_, dict)
                and event.metadata_.get('external_user_id')
                in {'first-' + handle, 'second-' + handle}
                for event in audit
            )
        issued_link = await management.invite(
            'http',
            'portal',
            'first-' + handle if isinstance(results[0], Ok) else 'second-' + handle,
            'next-bot',
        )
        linker = PrincipalLinker(database, GUIDManager(), ClockManager(), _TOKENS)
        linked = (
            await linker.link(issued_link.token, 'telegram', 'next-bot', handle)
        ).unwrap()
        assert (linked.user_id, linked.memory_scope) == (winner.user_id, handle)
    finally:
        await database.dispose()


async def test_operator_can_continue_an_unowned_existing_default_scope(
    session: AsyncSession,
) -> None:
    invitation = await RegistrationService(repo(session), _TOKENS).issue(
        'http',
        'portal',
        timedelta(minutes=10),
        'operator',
        existing_scope='default',
    )
    principal = (
        await RegistrationService(repo(session), _TOKENS).claim(
            'http',
            'portal',
            'recipient',
            name(),
            credential=invitation.token,
        )
    ).unwrap()
    assert principal.memory_scope == 'default'
    assert principal.handle != 'default'
