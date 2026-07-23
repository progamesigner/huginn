from collections.abc import Callable
from datetime import timedelta


class Timer:
    source: Callable[[], float]
    start: float

    def __init__(self, source: Callable[[], float]) -> None:
        self.source = source
        self.start = source()

    def elapsed(self) -> timedelta:
        return timedelta(seconds=self.source() - self.start)
