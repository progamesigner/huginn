import asyncio
from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.features.identities.repositories import IdentityRepository
from app.features.identities.services import IdentityLinkingService
from app.features.identities.sources.sqlalchemy import (
    ExternalIdentityRecord,
    IdentityAuditEventRecord,
    IdentityLinkInviteRecord,
    UserRecord,
)
from app.features.identities.tokens import IdentityTokenCodec
from lib.guid import GUIDManager
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration

_SECRET = b'x' * 32


def service(session: AsyncSession) -> IdentityLinkingService:
    return IdentityLinkingService(
        IdentityRepository(session, GUIDManager(), ClockManager()),
        IdentityTokenCodec(_SECRET),
        ClockManager(),
    )


async def test_invite_stores_only_keyed_digest(session: AsyncSession) -> None:
    repository = IdentityRepository(session, GUIDManager(), ClockManager())
    user = await repository.create_user(f'user-{uuid4().hex}', 'Example User')
    linking = service(session)

    invite = await linking.issue_invite(
        user.id,
        'telegram',
        f'realm-{uuid4().hex}',
        timedelta(minutes=10),
        'operator',
    )
    record = (
        await session.execute(
            select(IdentityLinkInviteRecord).where(
                IdentityLinkInviteRecord.user_id == user.id,
            ),
        )
    ).scalar_one()

    assert invite.token.encode() not in record.token_digest
    assert invite.short_code.encode() not in record.token_digest
    assert linking.tokens.matches(record.token_digest, invite.token)
    assert linking.tokens.matches(record.token_digest, invite.short_code)


@pytest.mark.usefixtures('_migrated')
async def test_concurrent_completion_creates_one_link_and_success_event(
    database_url: str,
) -> None:
    engine = create_async_engine(database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    realm = f'realm-{uuid4().hex}'
    handle = f'user-{uuid4().hex}'

    try:
        async with sessions() as setup:
            repository = IdentityRepository(setup, GUIDManager(), ClockManager())
            user = await repository.create_user(handle, 'Example User')
            linking = service(setup)
            invite = await linking.issue_invite(
                user.id,
                'telegram',
                realm,
                timedelta(minutes=10),
                'operator',
            )
            await setup.commit()

        async def complete() -> object:
            async with sessions() as current:
                linked = await service(current).complete(
                    invite.token,
                    'telegram',
                    realm,
                    '42',
                )
                await current.commit()
                return linked.unwrap().id

        first, second = await asyncio.gather(complete(), complete())

        async with sessions() as verification:
            links = await verification.scalar(
                select(func.count())
                .select_from(ExternalIdentityRecord)
                .where(
                    ExternalIdentityRecord.provider == 'telegram',
                    ExternalIdentityRecord.realm == realm,
                    ExternalIdentityRecord.external_user_id == '42',
                ),
            )
            successes = await verification.scalar(
                select(func.count())
                .select_from(IdentityAuditEventRecord)
                .where(
                    IdentityAuditEventRecord.event_type == 'link_succeeded',
                    IdentityAuditEventRecord.realm == realm,
                ),
            )

            assert first == second
            assert links == 1
            assert successes == 1

            await verification.execute(
                delete(IdentityAuditEventRecord).where(
                    IdentityAuditEventRecord.realm == realm,
                ),
            )
            await verification.execute(
                delete(UserRecord).where(UserRecord.handle == handle),
            )
            await verification.commit()
    finally:
        await engine.dispose()


async def test_a_rolled_back_link_keeps_none_of_its_audit(
    session: AsyncSession,
) -> None:
    repository = IdentityRepository(session, GUIDManager(), ClockManager())
    user = await repository.create_user(f'user-{uuid4().hex}', 'Example User')
    realm = f'realm-{uuid4().hex}'
    linking = service(session)
    invite = await linking.issue_invite(
        user.id,
        'telegram',
        realm,
        timedelta(minutes=10),
        'operator',
    )

    await linking.complete(invite.token, 'telegram', realm, '42')
    await session.rollback()

    # NOTE: the success audit has to live and die with the link it describes -- writing
    # it in a transaction of its own (the tempting way to keep failure audits alive)
    # would leave 'link_succeeded' behind for a link nobody ever made
    events = await session.scalar(
        select(func.count())
        .select_from(IdentityAuditEventRecord)
        .where(IdentityAuditEventRecord.realm == realm),
    )
    links = await session.scalar(
        select(func.count())
        .select_from(ExternalIdentityRecord)
        .where(ExternalIdentityRecord.realm == realm),
    )

    assert (events, links) == (0, 0)


async def test_invite_cannot_be_consumed_and_revoked(session: AsyncSession) -> None:
    repository = IdentityRepository(session, GUIDManager(), ClockManager())
    user = await repository.create_user(f'user-{uuid4().hex}', 'Example User')
    now = service(session).clock.now()
    invite = await repository.issue_invite(
        user.id,
        'telegram',
        f'realm-{uuid4().hex}',
        b'x' * 32,
        now + timedelta(minutes=10),
        'operator',
    )
    await repository.consume_invite(invite.id, now)

    with pytest.raises(IntegrityError):
        await repository.revoke_invite(invite.id, now)
