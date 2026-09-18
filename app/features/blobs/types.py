from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

type BlobKind = Literal['audio', 'document', 'image', 'video']


@dataclass(frozen=True)
class Blob:
    id: UUID
    sha256: str
    mime: str
    size: int
    kind: BlobKind
    filename: str | None = None
    expires_at: datetime | None = None
    data: bytes | None = None
