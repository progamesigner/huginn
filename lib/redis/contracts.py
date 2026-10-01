from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from redis.asyncio import Redis


class RedisContract(ABC):
    @abstractmethod
    def connect(
        self,
        alias: str | None = None,
    ) -> AbstractAsyncContextManager[Redis[bytes]]: ...
