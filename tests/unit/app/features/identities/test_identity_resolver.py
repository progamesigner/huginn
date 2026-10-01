from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

from app.features.identities.entities import AnonymousPrincipal, LinkedPrincipal
from app.features.identities.services import IdentityResolver

_USER_ID = UUID(int=1)
_IDENTITY_ID = UUID(int=2)


async def test_resolve_returns_linked_principal_for_known_identity() -> None:
    repository = MagicMock()
    repository.find_linked = AsyncMock(
        return_value=LinkedPrincipal(
            user_id=_USER_ID,
            external_identity_id=_IDENTITY_ID,
            handle='alice',
        ),
    )

    principal = await IdentityResolver(repository).resolve(
        'telegram',
        'primary',
        '42',
    )

    assert principal == LinkedPrincipal(
        user_id=_USER_ID,
        external_identity_id=_IDENTITY_ID,
        handle='alice',
    )
    assert principal.stable_key == f'user:{_USER_ID}'


async def test_resolve_keeps_unknown_identity_as_anonymous() -> None:
    repository = MagicMock()
    repository.find_linked = AsyncMock(return_value=None)

    principal = await IdentityResolver(repository).resolve(
        'telegram',
        'primary',
        '42',
    )

    assert principal == AnonymousPrincipal(
        provider='telegram',
        realm='primary',
        external_user_id='42',
    )


def test_anonymous_principal_keys_do_not_collide_at_part_boundaries() -> None:
    first = AnonymousPrincipal(provider='ab', realm='c', external_user_id='d')
    second = AnonymousPrincipal(provider='a', realm='bc', external_user_id='d')

    assert first.stable_key != second.stable_key
