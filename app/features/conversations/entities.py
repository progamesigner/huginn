from collections.abc import Mapping
from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import Field, JsonValue

from app.core.schema import AppEntity, AppValue


class Conversation(AppEntity):
    id: Annotated[UUID, Field()]
    channel: Annotated[str, Field()]
    chat_id: Annotated[str, Field()]
    created_at: Annotated[datetime | None, Field()] = None
    updated_at: Annotated[datetime | None, Field()] = None
    compact_version: Annotated[int, Field()] = 0
    summarized_through_turn_id: Annotated[UUID | None, Field()] = None
    reset_through_turn_id: Annotated[UUID | None, Field()] = None


class Part(AppValue):
    role: Annotated[str, Field()]
    kind: Annotated[str, Field()]
    payload: Annotated[Mapping[str, JsonValue], Field()] = Field(default_factory=dict)


class Turn(AppEntity):
    id: Annotated[UUID, Field()]
    agent: Annotated[str, Field()]
    parts: Annotated[tuple[Part, ...], Field()] = ()
    usage: Annotated[Mapping[str, JsonValue], Field()] = Field(default_factory=dict)
    created_at: Annotated[datetime | None, Field()] = None
