from collections.abc import Awaitable, Callable
from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.features.identities.exceptions import InvalidIdentityLinkError
from app.features.identities.repositories import IdentityRepository
from app.features.identities.schemas.sqlalchemy import IdentityAuditEventRecord
from app.features.identities.services import (
    IdentityLinkingService,
    PrincipalLinker,
    PrincipalResolver,
)
from app.features.identities.tokens import IdentityTokenCodec
from app.features.identities.types import AnonymousPrincipal, LinkedPrincipal
from lib.database import DatabaseContract, DatabaseManager
from lib.guid import GUIDManager
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration

_SECRET = b'x' * 32


def _linker(database: DatabaseContract) -> PrincipalLinker:
    return PrincipalLinker(
        database,
        GUIDManager(),
        ClockManager(),
        IdentityTokenCodec(_SECRET),
    )


def _resolver(database: DatabaseContract) -> PrincipalResolver:
    return PrincipalResolver(database, GUIDManager(), ClockManager())


async def _invite(database: DatabaseContract, realm: str) -> tuple[str, str]:
    async with database.session() as session, session.begin():
        repository = IdentityRepository(session, GUIDManager(), ClockManager())
        user = await repository.create_user(f'user-{uuid4().hex}', 'Example User')
        issued = await IdentityLinkingService(
            repository,
            IdentityTokenCodec(_SECRET),
            ClockManager(),
        ).issue_invite(
            user.id,
            'telegram',
            realm,
            timedelta(minutes=10),
            'operator',
        )

    return issued.short_code, str(user.id)


async def _prepared(
    database: DatabaseContract,
    realm: str,
    *,
    expires_in: timedelta = timedelta(minutes=10),
    revoked: bool = False,
    consumed: bool = False,
) -> str:
    # NOTE: straight to the repository -- the service refuses to issue the stale and
    # spent invites these cases are about
    clock = ClockManager()
    _, short_code, digest = IdentityTokenCodec(_SECRET).issue()
    async with database.session() as session, session.begin():
        repository = IdentityRepository(session, GUIDManager(), clock)
        user = await repository.create_user(f'user-{uuid4().hex}', 'Example User')
        invite = await repository.issue_invite(
            user.id,
            'telegram',
            realm,
            digest,
            clock.now() + expires_in,
            'operator',
        )
        if revoked:
            await repository.revoke_invite(invite.id, clock.now())
        if consumed:
            await repository.consume_invite(invite.id, clock.now())

    return short_code


async def _never_issued(database: DatabaseContract, realm: str, chat: str) -> str:
    return 'NOPE-NOPE-0000'


async def _for_another_realm(database: DatabaseContract, realm: str, chat: str) -> str:
    return await _prepared(database, f'other-{realm}')


async def _already_expired(database: DatabaseContract, realm: str, chat: str) -> str:
    return await _prepared(database, realm, expires_in=-timedelta(minutes=1))


async def _revoked_by_an_operator(
    database: DatabaseContract,
    realm: str,
    chat: str,
) -> str:
    return await _prepared(database, realm, revoked=True)


async def _spent_on_another_chat(
    database: DatabaseContract,
    realm: str,
    chat: str,
) -> str:
    return await _prepared(database, realm, consumed=True)


async def _aimed_at_a_taken_chat(
    database: DatabaseContract,
    realm: str,
    chat: str,
) -> str:
    # NOTE: the chat already answers to one account, so a second account's invite has
    # nothing left to claim
    await _linker(database).link(
        await _prepared(database, realm),
        'telegram',
        realm,
        chat,
    )

    return await _prepared(database, realm)


async def _failures(database: DatabaseContract, realm: str) -> list[object]:
    # NOTE: a session of its own -- the whole point is that the row outlived the
    # transaction the rejection rolled back
    async with database.session() as session:
        query = select(IdentityAuditEventRecord).where(
            IdentityAuditEventRecord.realm == realm,
            IdentityAuditEventRecord.event_type == 'link_failed',
        )
        records = (await session.execute(query)).scalars().all()

    return [record.metadata_ for record in records]


@pytest.mark.parametrize(
    ('prepare', 'reason'),
    [
        (_never_issued, 'unknown_token'),
        (_for_another_realm, 'realm_mismatch'),
        (_already_expired, 'expired'),
        (_revoked_by_an_operator, 'revoked'),
        (_spent_on_another_chat, 'consumed'),
        (_aimed_at_a_taken_chat, 'identity_conflict'),
    ],
)
@pytest.mark.usefixtures('_migrated')
async def test_a_refused_attempt_leaves_a_trail(
    prepare: Callable[[DatabaseContract, str, str], Awaitable[str]],
    reason: str,
    database_url: str,
) -> None:
    database = DatabaseManager({None: database_url})
    realm = f'realm-{uuid4().hex}'
    chat = uuid4().hex
    try:
        credential = await prepare(database, realm, chat)

        with pytest.raises(InvalidIdentityLinkError):
            await _linker(database).link(credential, 'telegram', realm, chat)

        assert await _failures(database, realm) == [
            {'reason': reason, 'external_user_id': chat},
        ]
    finally:
        await database.dispose()


@pytest.mark.usefixtures('_migrated')
async def test_a_link_that_goes_through_records_no_failure(
    database_url: str,
) -> None:
    database = DatabaseManager({None: database_url})
    realm = f'realm-{uuid4().hex}'
    try:
        code, _ = await _invite(database, realm)
        await _linker(database).link(code, 'telegram', realm, uuid4().hex)

        assert await _failures(database, realm) == []
    finally:
        await database.dispose()


@pytest.mark.usefixtures('_migrated')
async def test_linking_a_chat_commits_and_the_resolver_sees_it(
    database_url: str,
) -> None:
    database = DatabaseManager({None: database_url})
    realm = f'realm-{uuid4().hex}'
    external = uuid4().hex
    try:
        code, user_id = await _invite(database, realm)

        # NOTE: this is the command path -- the contract owns its own transaction, so
        # a link made here has to be visible to the next round's resolver
        principal = await _linker(database).link(code, 'telegram', realm, external)
        resolved = await _resolver(database).resolve('telegram', realm, external)

        assert isinstance(principal, LinkedPrincipal)
        assert str(principal.user_id) == user_id
        assert resolved == principal
    finally:
        await database.dispose()


@pytest.mark.usefixtures('_migrated')
async def test_a_code_nobody_issued_leaves_the_chat_anonymous(
    database_url: str,
) -> None:
    database = DatabaseManager({None: database_url})
    realm = f'realm-{uuid4().hex}'
    external = uuid4().hex
    try:
        with pytest.raises(InvalidIdentityLinkError):
            await _linker(database).link('NOPE-NOPE-0000', 'telegram', realm, external)

        resolved = await _resolver(database).resolve('telegram', realm, external)

        assert isinstance(resolved, AnonymousPrincipal)
    finally:
        await database.dispose()
