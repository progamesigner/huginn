from datetime import timedelta
from uuid import uuid4

import pytest

from app.features.identities.exceptions import InvalidIdentityLinkError
from app.features.identities.repositories import IdentityRepository
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
