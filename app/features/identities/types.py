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
