from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class WebhookStatus:
    url: str
    pending: int
    certificate: bool
    allowed: tuple[str, ...]
    connections: int | None = None
    failed_at: datetime | None = None
    failure: str | None = None
