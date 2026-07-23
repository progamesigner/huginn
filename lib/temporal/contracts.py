from abc import ABC, abstractmethod
from datetime import datetime

from .timer import Timer


class ClockContract(ABC):
    @abstractmethod
    def now(self) -> datetime: ...

    @abstractmethod
    def timer(self) -> Timer: ...
