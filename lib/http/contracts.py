from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager

from httpx2 import URL, AsyncClient


class HTTPClientContract(ABC):
    @abstractmethod
    def connect(
        self,
        base_url: URL | str = '',
        headers: Mapping[str, str] | None = None,
    ) -> AbstractAsyncContextManager[AsyncClient]: ...
