import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.identities.entities import LinkedPrincipal
from app.features.identities.repositories import IdentityRepository
from lib.guid import GUIDManager
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration


def repository(session: AsyncSession) -> IdentityRepository:
    return IdentityRepository(session, GUIDManager(), ClockManager())


async def test_user_handle_is_normalized_and_globally_unique(
    session: AsyncSession,
) -> None:
    identities = repository(session)
    user = await identities.create_user(' Alice ', 'Alice Example')

    assert user.handle == 'alice'
    assert await identities.find_user('ALICE') == user

    with pytest.raises(IntegrityError):
        await identities.create_user('alice', 'Someone else')


async def test_external_identity_is_unique_within_provider_and_realm(
    session: AsyncSession,
) -> None:
    identities = repository(session)
    first = await identities.create_user('first', 'First')
    second = await identities.create_user('second', 'Second')
    linked = await identities.link(
        first.id,
        'telegram',
        'primary',
        '42',
        {'username': 'alice'},
    )

    assert await identities.find_external('telegram', 'primary', '42') == linked
    assert await identities.find_external('telegram', 'secondary', '42') is None
    assert dict(linked.profile) == {'username': 'alice'}

    with pytest.raises(IntegrityError):
        await identities.link(second.id, 'telegram', 'primary', '42')


async def test_user_can_link_multiple_external_identities(
    session: AsyncSession,
) -> None:
    identities = repository(session)
    user = await identities.create_user('alice', 'Alice')

    telegram = await identities.link(user.id, 'telegram', 'primary', '42')
    slack = await identities.link(user.id, 'slack', 'swag', 'U42')

    assert await identities.list_external(user.id) == (slack, telegram)


async def test_unlink_removes_mapping_without_losing_audit_reference(
    session: AsyncSession,
) -> None:
    identities = repository(session)
    user = await identities.create_user('alice', 'Alice')
    linked = await identities.link(user.id, 'telegram', 'primary', '42')
    event = await identities.record_audit(
        'link_succeeded',
        'telegram',
        'primary',
        user_id=user.id,
        external_identity_id=linked.id,
        metadata={'source': 'test'},
    )

    assert await identities.unlink(linked.id)
    assert await identities.find_external('telegram', 'primary', '42') is None
    assert event.external_identity_id == linked.id
    assert event.occurred_at is not None
    assert dict(event.metadata) == {'source': 'test'}


async def test_find_linked_reads_the_principal_with_its_handle(
    session: AsyncSession,
) -> None:
    identities = repository(session)
    user = await identities.create_user('Carol', 'Carol')
    linked = await identities.link(user.id, 'telegram', 'primary', '77')

    assert await identities.find_linked('telegram', 'primary', '77') == (
        LinkedPrincipal(user_id=user.id, external_identity_id=linked.id, handle='carol')
    )
    assert await identities.find_linked('telegram', 'primary', '78') is None
