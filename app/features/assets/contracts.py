from abc import ABC, abstractmethod
from collections.abc import Mapping

from pydantic import JsonValue

from .entities import Asset, AssetDocument, AssetKind, AssetMeta


class AssetsServiceContract(ABC):
    @abstractmethod
    async def inventory(
        self,
        source: str,
        kind: AssetKind,
        scope: Mapping[str, JsonValue],
    ) -> tuple[AssetMeta, ...]: ...

    @abstractmethod
    async def load(
        self,
        source: str,
        kind: AssetKind,
        name: str,
        scope: Mapping[str, JsonValue],
    ) -> Asset | None: ...

    @abstractmethod
    async def read_one(
        self,
        source: str,
        path: str,
        scope: Mapping[str, JsonValue],
    ) -> AssetDocument | None: ...

    @abstractmethod
    def invalidate(self, source: str | None = None) -> None: ...
