from abc import ABC, abstractmethod
from collections.abc import Coroutine
from datetime import timedelta
from typing import Any


class TaskSupervisorContract(ABC):
    @abstractmethod
    async def dispose(self) -> None: ...

    @abstractmethod
    async def drain(self, grace: timedelta | None = None) -> bool: ...

    @abstractmethod
    def spawn(self, coro: Coroutine[Any, Any, None]) -> None: ...
