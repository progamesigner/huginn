from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from redis.asyncio import Redis


class RedisContract(ABC):
    @abstractmethod
    def connect(
        self,
        instance: str | None = None,
    ) -> AbstractAsyncContextManager[Redis[bytes]]: ...
