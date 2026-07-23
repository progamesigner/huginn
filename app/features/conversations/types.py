from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from pydantic import JsonValue


@dataclass(frozen=True)
class Conversation:
    id: UUID
    channel: str
    chat_id: str
    agent: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    compact_version: int = 0
    summarized_through_turn_id: UUID | None = None


@dataclass(frozen=True)
class Part:
    role: str
    kind: str
    payload: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Turn:
    id: UUID
    parts: tuple[Part, ...] = ()
    usage: Mapping[str, JsonValue] = field(default_factory=dict)
    created_at: datetime | None = None
