from datetime import UTC, datetime, timedelta
from typing import override

from lib.temporal import ClockContract, Timer

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class FakeClock(ClockContract):
    instant: datetime
    monotonic: float

    def __init__(self, now: datetime = _EPOCH) -> None:
        self.instant = now
        self.monotonic = 0.0

    @override
    def now(self) -> datetime:
        return self.instant

    @override
    def timer(self) -> Timer:
        return Timer(lambda: self.monotonic)

    def advance(self, delta: timedelta) -> None:
        self.instant += delta
        self.monotonic += delta.total_seconds()
