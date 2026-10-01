import builtins
from collections.abc import Mapping
from datetime import timedelta
from typing import override

from pydantic import JsonValue

from app.features.skills.entities import Skill, SkillMeta
from app.features.skills.repositories import SkillRepository
from app.features.skills.sources.contracts import SkillSource
from tests.helpers.clock import FakeClock


class FakeSource(SkillSource):
    metas: tuple[SkillMeta, ...]
    list_calls: int

    def __init__(self, metas: tuple[SkillMeta, ...]) -> None:
        self.metas = metas
        self.list_calls = 0

    @override
    async def list(self, scope: Mapping[str, JsonValue]) -> tuple[SkillMeta, ...]:
        self.list_calls += 1
        return self.metas

    @override
    async def load(self, path: str, scope: Mapping[str, JsonValue]) -> Skill:
        meta = next(meta for meta in self.metas if meta.path == path)
        return Skill(meta=meta, body=f'body of {meta.name}')


def _meta(name: str) -> SkillMeta:
    return SkillMeta(name=name, description=f'{name} desc', path=f'skills/{name}.md')


def repository() -> tuple[SkillRepository, FakeSource, FakeClock]:
    source = FakeSource((_meta('alpha'), _meta('beta')))
    clock = FakeClock()
    return SkillRepository(source, clock, ttl=timedelta(minutes=15)), source, clock


async def test_inventory_is_cached_within_ttl() -> None:
    repo, source, clock = repository()

    await repo.inventory({})
    clock.advance(timedelta(minutes=14))
    await repo.inventory({})

    assert source.list_calls == 1


async def test_inventory_refreshes_after_ttl() -> None:
    repo, source, clock = repository()

    await repo.inventory({})
    clock.advance(timedelta(minutes=15))
    await repo.inventory({})

    assert source.list_calls == 2
    assert len(repo.cache) == 1


async def test_load_finds_skill_by_name() -> None:
    repo, _, _ = repository()

    skill = await repo.load('beta', {})

    assert skill is not None
    assert skill.body == 'body of beta'


async def test_load_returns_none_for_unknown_skill() -> None:
    repo, _, _ = repository()

    assert await repo.load('missing', {}) is None


class ScopedSource(SkillSource):
    calls: builtins.list[dict[str, JsonValue]]

    def __init__(self) -> None:
        self.calls = []

    @override
    async def list(
        self,
        scope: Mapping[str, JsonValue],
    ) -> tuple[SkillMeta, ...]:
        copied = dict(scope)
        self.calls.append(copied)
        return (_meta(str(copied['user'])),)

    @override
    async def load(self, path: str, scope: Mapping[str, JsonValue]) -> Skill:
        return Skill(meta=_meta(str(scope['user'])), body=path)


async def test_inventory_cache_is_isolated_by_scope() -> None:
    source = ScopedSource()
    repo = SkillRepository(source, FakeClock(), ttl=timedelta(minutes=15))

    alice = await repo.inventory({'user': 'alice'})
    bob = await repo.inventory({'user': 'bob'})
    again = await repo.inventory({'user': 'alice'})

    assert alice[0].name == 'alice'
    assert bob[0].name == 'bob'
    assert again == alice
    assert source.calls == [{'user': 'alice'}, {'user': 'bob'}]
