from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from app.core.schema import AppEntity

type BlobKind = Literal['audio', 'document', 'image', 'video']


class Blob(AppEntity):
    id: Annotated[UUID, Field()]
    sha256: Annotated[str, Field()]
    mime: Annotated[str, Field()]
    size: Annotated[int, Field()]
    kind: Annotated[BlobKind, Field()]
    filename: Annotated[str | None, Field()] = None
    expires_at: Annotated[datetime | None, Field()] = None
    data: Annotated[bytes | None, Field()] = None


# NOTE: the name a file arrived under is its sender's, so it lives on the grant
# rather than on the blob every sender of the same bytes shares
class BlobGrant(AppEntity):
    blob_id: Annotated[UUID, Field()]
    owner: Annotated[str, Field()]
    filename: Annotated[str | None, Field()] = None
