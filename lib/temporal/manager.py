import time
from datetime import UTC, datetime
from typing import override

from .contracts import ClockContract
from .timer import Timer


class ClockManager(ClockContract):
    @override
    def now(self) -> datetime:
        return datetime.now(UTC)

    @override
    def timer(self) -> Timer:
        return Timer(time.monotonic)
