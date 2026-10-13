import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import JsonValue

from lib.temporal import ClockContract

from .entities import Skill, SkillMeta
from .sources.contracts import SkillSource

_DEFAULT_TTL = timedelta(minutes=15)

type _ScopeKey = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _Inventory:
    expires_at: datetime
    skills: tuple[SkillMeta, ...]


class SkillRepository:
    source: SkillSource
    clock: ClockContract
    ttl: timedelta
    cache: dict[_ScopeKey, _Inventory]

    def __init__(
        self,
        source: SkillSource,
        clock: ClockContract,
        ttl: timedelta = _DEFAULT_TTL,
    ) -> None:
        self.source = source
        self.clock = clock
        self.ttl = ttl
        self.cache = {}

    async def inventory(
        self,
        scope: Mapping[str, JsonValue],
    ) -> tuple[SkillMeta, ...]:
        now = self.clock.now()
        self.cache = {
            key: cached for key, cached in self.cache.items() if cached.expires_at > now
        }
        key = tuple(
            sorted(
                (name, json.dumps(value, sort_keys=True))
                for name, value in scope.items()
            ),
        )
        cached = self.cache.get(key)
        if cached is None:
            cached = _Inventory(now + self.ttl, await self.source.list(scope))
            self.cache[key] = cached
        return cached.skills

    async def load(
        self,
        name: str,
        scope: Mapping[str, JsonValue],
    ) -> Skill | None:
        meta = next(
            (meta for meta in await self.inventory(scope) if meta.name == name),
            None,
        )
        # NOTE: full body is read fresh (uncached) so an edited skill takes effect
        return await self.source.load(meta.path, scope) if meta is not None else None


class SkillRepositories:
    def __init__(self, factory: Callable[[str], SkillRepository]) -> None:
        self.factory = factory
        self.repositories: dict[str, SkillRepository] = {}

    def get(self, name: str) -> SkillRepository:
        if name not in self.repositories:
            self.repositories[name] = self.factory(name)
        return self.repositories[name]
