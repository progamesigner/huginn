import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import override

from pydantic import JsonValue

from lib.mcp import MCPClientContract
from lib.observability import Logger
from lib.temporal import ClockContract

from .config import AssetsConfig
from .contracts import AssetsServiceContract
from .entities import Asset, AssetDocument, AssetKind, AssetMeta
from .exceptions import AssetError
from .sources import AssetSource, parse

type CacheKey = tuple[str, AssetKind, str]


@dataclass(frozen=True)
class Inventory:
    expires: datetime
    items: tuple[AssetMeta, ...]


@dataclass
class Refresh:
    task: asyncio.Task[tuple[AssetMeta, ...]]
    waiters: int = 0


class AssetsService(AssetsServiceContract):
    def __init__(
        self,
        config: AssetsConfig,
        client: MCPClientContract | None,
        clock: ClockContract,
        *,
        logger: Logger,
        ttl: timedelta = timedelta(minutes=15),
    ) -> None:
        self.config = config
        self.clock = clock
        self.ttl = ttl
        self.sources = {
            name: AssetSource(name, source, client, logger=logger)
            for name, source in config.sources.items()
        }
        self.cache: dict[CacheKey, Inventory] = {}
        self.pending: dict[CacheKey, Refresh] = {}
        self.generation = 0

    @override
    def invalidate(self, source: str | None = None) -> None:
        self.generation += 1
        self.cache = {
            key: value
            for key, value in self.cache.items()
            if source is not None and key[0] != source
        }
        # Old in-flight callers may finish, but subsequent callers must refresh.
        self.pending = {
            key: value
            for key, value in self.pending.items()
            if source is not None and key[0] != source
        }

    async def _refresh(
        self,
        key: CacheKey,
        scope: Mapping[str, JsonValue],
        generation: int,
    ) -> tuple[AssetMeta, ...]:
        items = await self.sources[key[0]].inventory(key[1], scope)
        if generation == self.generation and len(self.cache) < self.config.max_entries:
            self.cache[key] = Inventory(self.clock.now() + self.ttl, items)
        return items

    @override
    async def inventory(
        self,
        source: str,
        kind: AssetKind,
        scope: Mapping[str, JsonValue],
    ) -> tuple[AssetMeta, ...]:
        self.config.resolve_source(source)
        now = self.clock.now()
        self.cache = {
            key: value for key, value in self.cache.items() if value.expires > now
        }
        key = (
            source,
            kind,
            json.dumps(dict(scope), sort_keys=True, separators=(',', ':')),
        )
        if cached := self.cache.get(key):
            return cached.items
        refresh = self.pending.get(key)
        if refresh is None:
            # At capacity reads bypass caching; the in-flight table is bounded too.
            if len(self.pending) >= self.config.max_entries:
                return await self.sources[source].inventory(kind, scope)
            task = asyncio.create_task(self._refresh(key, dict(scope), self.generation))
            refresh = Refresh(task)
            self.pending[key] = refresh
        refresh.waiters += 1
        try:
            return await asyncio.shield(refresh.task)
        finally:
            refresh.waiters -= 1
            if not refresh.waiters:
                if self.pending.get(key) is refresh:
                    del self.pending[key]
                if not refresh.task.done():
                    refresh.task.cancel()
                await asyncio.gather(refresh.task, return_exceptions=True)

    @override
    async def read_one(
        self,
        source: str,
        path: str,
        scope: Mapping[str, JsonValue],
    ) -> AssetDocument | None:
        self.config.resolve_source(source)
        return await self.sources[source].read(path, scope)

    @override
    async def load(
        self,
        source: str,
        kind: AssetKind,
        name: str,
        scope: Mapping[str, JsonValue],
    ) -> Asset | None:
        meta = next(
            (
                item
                for item in await self.inventory(source, kind, scope)
                if item.name == name
            ),
            None,
        )
        if meta is None:
            return None
        document = await self.read_one(source, meta.path, scope)
        if document is None:
            return None
        asset = parse(source, kind, meta.path, document.content)
        if asset.meta.name != name:
            raise AssetError('asset name changed; refresh the inventory')
        return asset
