import hashlib
from datetime import UTC, datetime

import pytest

from app.features.identities.tokens import IdentityTokenCodec
from app.features.identities.types import IssuedIdentityLinkInvite

_SECRET = b'x' * 32
_RAW = bytes(range(16))


def test_deep_token_and_short_code_share_one_digest() -> None:
    codec = IdentityTokenCodec(_SECRET, random_bytes=lambda _: _RAW)

    token, short_code, digest = codec.issue()

    assert codec.digest_credential(token) == digest
    assert codec.digest_credential(short_code.lower()) == digest
    assert codec.matches(digest, token)


def test_short_code_rejects_a_bad_checksum() -> None:
    codec = IdentityTokenCodec(_SECRET, random_bytes=lambda _: _RAW)
    _, short_code, _ = codec.issue()

    assert codec.digest_credential(f'{short_code[:-1]}0') is None


def test_digest_is_keyed_and_plain_token_is_not_stored() -> None:
    codec = IdentityTokenCodec(_SECRET, random_bytes=lambda _: _RAW)
    token, _, digest = codec.issue()

    assert digest != hashlib.sha256(_RAW).digest()
    assert token.encode() not in digest


def test_secret_requires_256_bits() -> None:
    with pytest.raises(ValueError, match='at least 32 bytes'):
        IdentityTokenCodec(b'too-short')


def test_invite_repr_redacts_both_credentials() -> None:
    invite = IssuedIdentityLinkInvite(
        'deep-secret', 'short-secret', expires_at=datetime(2026, 1, 1, tzinfo=UTC)
    )

    assert 'deep-secret' not in repr(invite)
    assert 'short-secret' not in repr(invite)
