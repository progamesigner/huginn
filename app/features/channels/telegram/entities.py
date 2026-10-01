from datetime import datetime
from typing import Annotated

from pydantic import Field

from app.core.schema import AppValue


class WebhookStatus(AppValue):
    url: Annotated[str, Field()]
    pending: Annotated[int, Field()]
    certificate: Annotated[bool, Field()]
    allowed: Annotated[tuple[str, ...], Field()]
    connections: Annotated[int | None, Field()] = None
    failed_at: Annotated[datetime | None, Field()] = None
    failure: Annotated[str | None, Field()] = None
