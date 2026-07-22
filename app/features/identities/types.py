from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from pydantic import JsonValue


@dataclass(frozen=True)
class User:
    id: UUID
    handle: str
    display_name: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class ExternalIdentity:
    id: UUID
    user_id: UUID
    provider: str
    realm: str
    external_user_id: str
    profile: Mapping[str, JsonValue] = field(default_factory=dict)
    linked_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class IdentityAuditEvent:
    id: UUID
    event_type: str
    provider: str
    realm: str
    user_id: UUID | None = None
    external_identity_id: UUID | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    occurred_at: datetime | None = None


@dataclass(frozen=True)
class IdentityLinkInvite:
    id: UUID
    user_id: UUID
    provider: str
    realm: str
    expires_at: datetime
    created_by: str
    consumed_at: datetime | None = None
    revoked_at: datetime | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class LinkedPrincipal:
    user_id: UUID
    external_identity_id: UUID

    @property
    def stable_key(self) -> str:
        return f'user:{self.user_id}'


@dataclass(frozen=True)
class AnonymousPrincipal:
    provider: str
    realm: str
    external_user_id: str

    @property
    def stable_key(self) -> str:
        parts = (self.provider, self.realm, self.external_user_id)
        encoded = ':'.join(f'{len(part)}:{part}' for part in parts)
        return f'anonymous:{encoded}'


type Principal = LinkedPrincipal | AnonymousPrincipal


@dataclass(frozen=True)
class IssuedIdentityLinkInvite:
    token: str = field(repr=False)
    short_code: str = field(repr=False)
    expires_at: datetime
