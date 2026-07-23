from datetime import datetime, timedelta

from lib.temporal import ClockContract

from .sources.contracts import SkillSource
from .types import Skill, SkillMeta

_DEFAULT_TTL = timedelta(minutes=15)


class SkillRepository:
    source: SkillSource
    clock: ClockContract
    ttl: timedelta
    cache: tuple[SkillMeta, ...] | None
    expires_at: datetime | None

    def __init__(
        self,
        source: SkillSource,
        clock: ClockContract,
        ttl: timedelta = _DEFAULT_TTL,
    ) -> None:
        self.source = source
        self.clock = clock
        self.ttl = ttl
        self.cache = None
        self.expires_at = None

    async def inventory(self) -> tuple[SkillMeta, ...]:
        now = self.clock.now()
        if self.cache is None or self.expires_at is None or now >= self.expires_at:
            self.cache = await self.source.list()
            self.expires_at = now + self.ttl
        return self.cache

    async def load(self, name: str) -> Skill | None:
        meta = next(
            (meta for meta in await self.inventory() if meta.name == name),
            None,
        )
        # NOTE: full body is read fresh (uncached) so an edited skill takes effect
        return await self.source.load(meta.path) if meta is not None else None
