from collections.abc import Mapping
from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import Field, JsonValue

from app.core.schema import AppEntity, AppValue


class User(AppEntity):
    id: Annotated[UUID, Field()]
    handle: Annotated[str, Field()]
    display_name: Annotated[str, Field()]
    created_at: Annotated[datetime | None, Field()] = None
    updated_at: Annotated[datetime | None, Field()] = None


class ExternalIdentity(AppEntity):
    id: Annotated[UUID, Field()]
    user_id: Annotated[UUID, Field()]
    provider: Annotated[str, Field()]
    realm: Annotated[str, Field()]
    external_user_id: Annotated[str, Field()]
    profile: Annotated[Mapping[str, JsonValue], Field()] = Field(default_factory=dict)
    linked_at: Annotated[datetime | None, Field()] = None
    updated_at: Annotated[datetime | None, Field()] = None


class IdentityAuditEvent(AppEntity):
    id: Annotated[UUID, Field()]
    event_type: Annotated[str, Field()]
    provider: Annotated[str, Field()]
    realm: Annotated[str, Field()]
    user_id: Annotated[UUID | None, Field()] = None
    external_identity_id: Annotated[UUID | None, Field()] = None
    metadata: Annotated[Mapping[str, JsonValue], Field()] = Field(default_factory=dict)
    occurred_at: Annotated[datetime | None, Field()] = None


class IdentityLinkInvite(AppEntity):
    id: Annotated[UUID, Field()]
    user_id: Annotated[UUID, Field()]
    provider: Annotated[str, Field()]
    realm: Annotated[str, Field()]
    expires_at: Annotated[datetime, Field()]
    created_by: Annotated[str, Field()]
    consumed_at: Annotated[datetime | None, Field()] = None
    revoked_at: Annotated[datetime | None, Field()] = None
    created_at: Annotated[datetime | None, Field()] = None


class LinkedPrincipal(AppValue):
    user_id: Annotated[UUID, Field()]
    external_identity_id: Annotated[UUID, Field()]
    handle: Annotated[str, Field()]

    @property
    def stable_key(self) -> str:
        return f'user:{self.user_id}'


class AnonymousPrincipal(AppValue):
    provider: Annotated[str, Field()]
    realm: Annotated[str, Field()]
    external_user_id: Annotated[str, Field()]

    @property
    def stable_key(self) -> str:
        parts = (self.provider, self.realm, self.external_user_id)
        encoded = ':'.join(f'{len(part)}:{part}' for part in parts)
        return f'anonymous:{encoded}'


type Principal = LinkedPrincipal | AnonymousPrincipal


class IssuedIdentityLinkInvite(AppValue):
    token: Annotated[str, Field(repr=False)]
    short_code: Annotated[str, Field(repr=False)]
    expires_at: Annotated[datetime, Field()]
