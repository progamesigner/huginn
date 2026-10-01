from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from sqlalchemy.exc import IntegrityError

from app.features.identities.entities import (
    ExternalIdentity,
    IdentityLinkInvite,
)
from app.features.identities.exceptions import InvalidIdentityLinkError
from app.features.identities.services import IdentityLinkingService
from app.features.identities.tokens import IdentityTokenCodec
from tests.helpers.clock import FakeClock

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_SECRET = b'x' * 32
_USER_ID = UUID(int=1)
_INVITE_ID = UUID(int=2)
_IDENTITY_ID = UUID(int=3)


def repository() -> MagicMock:
    value = MagicMock()
    value.issue_invite = AsyncMock()
    value.record_audit = AsyncMock()
    value.lock_invite = AsyncMock(return_value=None)
    value.find_external = AsyncMock(return_value=None)
    value.link = AsyncMock()
    value.consume_invite = AsyncMock()
    value.lock_invite_by_id = AsyncMock()
    value.revoke_invite = AsyncMock()
    value.session.begin_nested.return_value.__aenter__ = AsyncMock()
    value.unlink = AsyncMock(return_value=True)
    value.session.begin_nested.return_value.__aexit__ = AsyncMock(return_value=False)
    return value


def stored_invite(**changes: object) -> IdentityLinkInvite:
    values = {
        'id': _INVITE_ID,
        'user_id': _USER_ID,
        'provider': 'telegram',
        'realm': 'primary',
        'expires_at': _NOW + timedelta(minutes=10),
        'created_by': 'operator',
    }
    values.update(changes)
    return IdentityLinkInvite(**cast(Any, values))


def service(repo: MagicMock) -> IdentityLinkingService:
    codec = IdentityTokenCodec(_SECRET, random_bytes=lambda _: bytes(range(16)))
    return IdentityLinkingService(repo, codec, clock=FakeClock(now=_NOW))


async def test_issue_invite_persists_only_digest_and_audits() -> None:
    repo = repository()
    repo.issue_invite.return_value = stored_invite()
    linking = service(repo)

    invite = await linking.issue_invite(
        _USER_ID,
        'telegram',
        'primary',
        timedelta(minutes=10),
        'operator',
    )

    digest = repo.issue_invite.await_args.args[3]
    assert isinstance(digest, bytes)
    assert invite.token.encode() not in digest
    assert invite.short_code.encode() not in digest
    repo.record_audit.assert_awaited_once()


async def test_complete_links_and_consumes_invite() -> None:
    repo = repository()
    linked = ExternalIdentity(
        id=_IDENTITY_ID,
        user_id=_USER_ID,
        provider='telegram',
        realm='primary',
        external_user_id='42',
    )
    repo.lock_invite.return_value = stored_invite()
    repo.link.return_value = linked
    linking = service(repo)
    token = linking.tokens.issue()[0]

    result = await linking.complete(token, 'telegram', 'primary', '42')

    assert result == linked
    repo.consume_invite.assert_awaited_once_with(_INVITE_ID, _NOW)
    assert repo.record_audit.await_args.args[0] == 'link_succeeded'


@pytest.mark.parametrize(
    ('changes', 'provider', 'realm', 'reason'),
    [
        ({'expires_at': _NOW}, 'telegram', 'primary', 'expired'),
        ({'revoked_at': _NOW}, 'telegram', 'primary', 'revoked'),
        ({}, 'telegram', 'other', 'realm_mismatch'),
    ],
)
async def test_complete_rejects_invalid_invite_with_one_public_error(
    changes: dict[str, datetime],
    provider: str,
    realm: str,
    reason: str,
) -> None:
    repo = repository()
    repo.lock_invite.return_value = stored_invite(**changes)
    linking = service(repo)
    token = linking.tokens.issue()[0]

    with pytest.raises(InvalidIdentityLinkError) as caught:
        await linking.complete(token, provider, realm, '42')

    assert str(caught.value) == 'This link is invalid or expired. Ask for a new invite.'
    assert token not in repr(caught.value)
    assert caught.value.reason == reason
    # NOTE: this service is handed a session it does not own, and the caller rolls that
    # session back on the way out -- an audit row written here would not survive, so the
    # reason travels on the exception and the caller records it
    repo.record_audit.assert_not_awaited()


async def test_a_consumed_invite_is_refused_for_another_chat() -> None:
    repo = repository()
    repo.lock_invite.return_value = stored_invite(consumed_at=_NOW)
    linking = service(repo)

    with pytest.raises(InvalidIdentityLinkError) as caught:
        await linking.complete(linking.tokens.issue()[0], 'telegram', 'primary', '42')

    assert caught.value.reason == 'consumed'
    repo.record_audit.assert_not_awaited()


async def test_a_chat_owned_by_another_account_is_refused() -> None:
    repo = repository()
    repo.lock_invite.return_value = stored_invite()
    repo.find_external.return_value = ExternalIdentity(
        id=_IDENTITY_ID,
        user_id=UUID(int=9),
        provider='telegram',
        realm='primary',
        external_user_id='42',
    )
    linking = service(repo)

    with pytest.raises(InvalidIdentityLinkError) as caught:
        await linking.complete(linking.tokens.issue()[0], 'telegram', 'primary', '42')

    assert caught.value.reason == 'identity_conflict'
    repo.link.assert_not_awaited()
    repo.record_audit.assert_not_awaited()


async def test_a_chat_claimed_by_a_racing_link_is_refused() -> None:
    repo = repository()
    repo.lock_invite.return_value = stored_invite()
    repo.link.side_effect = IntegrityError('insert', (), Exception('duplicate'))
    linking = service(repo)

    with pytest.raises(InvalidIdentityLinkError) as caught:
        await linking.complete(linking.tokens.issue()[0], 'telegram', 'primary', '42')

    assert caught.value.reason == 'identity_conflict'
    repo.consume_invite.assert_not_awaited()
    repo.record_audit.assert_not_awaited()


async def test_consumed_invite_is_idempotent_for_the_same_identity() -> None:
    repo = repository()
    linked = ExternalIdentity(
        id=_IDENTITY_ID,
        user_id=_USER_ID,
        provider='telegram',
        realm='primary',
        external_user_id='42',
    )
    repo.lock_invite.return_value = stored_invite(consumed_at=_NOW)
    repo.find_external.return_value = linked
    linking = service(repo)

    result = await linking.complete(
        linking.tokens.issue()[0],
        'telegram',
        'primary',
        '42',
    )

    assert result == linked
    repo.link.assert_not_awaited()


async def test_revoke_only_changes_an_active_invite() -> None:
    repo = repository()
    repo.lock_invite_by_id.return_value = stored_invite()
    linking = service(repo)

    assert await linking.revoke(_INVITE_ID)
    repo.revoke_invite.assert_awaited_once_with(_INVITE_ID, _NOW)
    assert repo.record_audit.await_args.args[0] == 'link_revoked'


async def test_unknown_credential_is_redacted_and_named_as_a_reason() -> None:
    repo = repository()
    linking = service(repo)
    credential = 'not-a-valid-credential'

    with pytest.raises(InvalidIdentityLinkError) as caught:
        await linking.complete(credential, 'telegram', 'primary', '42')

    assert credential not in repr(caught.value)
    assert caught.value.reason == 'unknown_token'
    repo.record_audit.assert_not_awaited()


async def test_issue_invite_rejects_non_positive_ttl() -> None:
    repo = repository()
    linking = service(repo)

    with pytest.raises(ValueError, match='ttl must be positive'):
        await linking.issue_invite(
            _USER_ID,
            'telegram',
            'primary',
            timedelta(),
            'operator',
        )
    repo.issue_invite.assert_not_awaited()


async def test_unlink_removes_identity_and_audits_revocation() -> None:
    repo = repository()
    linked = ExternalIdentity(
        id=_IDENTITY_ID,
        user_id=_USER_ID,
        provider='telegram',
        realm='primary',
        external_user_id='42',
    )
    linking = service(repo)

    assert await linking.unlink(linked)

    repo.unlink.assert_awaited_once_with(_IDENTITY_ID)
    repo.record_audit.assert_awaited_once_with(
        'link_revoked',
        'telegram',
        'primary',
        user_id=_USER_ID,
        external_identity_id=_IDENTITY_ID,
    )
