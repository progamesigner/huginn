from datetime import timedelta
from typing import override

from app.features.skills.repositories import SkillRepository
from app.features.skills.sources.contracts import SkillSource
from app.features.skills.tools import SkillParams, build_skill_tool
from app.features.skills.types import Skill, SkillMeta
from app.features.tools.registry import ToolSet
from lib.llm import ToolCallSegment
from tests.helpers.clock import FakeClock


class FakeSource(SkillSource):
    metas: tuple[SkillMeta, ...]

    def __init__(self, metas: tuple[SkillMeta, ...]) -> None:
        self.metas = metas

    @override
    async def list(self) -> tuple[SkillMeta, ...]:
        return self.metas

    @override
    async def load(self, path: str) -> Skill:
        meta = next(meta for meta in self.metas if meta.path == path)
        return Skill(meta=meta, body=f'# {meta.name}\n\ndo the thing')


def _meta(name: str) -> SkillMeta:
    return SkillMeta(name=name, description=f'{name} does things', path=f's/{name}.md')


async def build(*names: str) -> tuple[ToolSet, SkillRepository]:
    source = FakeSource(tuple(_meta(name) for name in names))
    repository = SkillRepository(source, FakeClock(), ttl=timedelta(minutes=15))
    inventory = await repository.inventory()
    tools = ToolSet()
    tools.add('skill', build_skill_tool(repository, inventory))
    return tools, repository


async def test_description_embeds_the_inventory() -> None:
    tools, _ = await build('alpha', 'beta')

    description = tools.tools['skill'].description

    assert "<skill name='alpha'>alpha does things</skill>" in description
    assert "<skill name='beta'>" in description


def test_schema_exposes_only_name() -> None:
    schema = SkillParams.model_json_schema()

    assert set(schema['properties']) == {'name'}


async def test_loading_a_skill_returns_its_body() -> None:
    tools, _ = await build('alpha')

    result = await tools.execute(
        ToolCallSegment(id='c1', name='skill', arguments='{"name": "alpha"}'),
    )

    assert not result.error
    assert 'do the thing' in result.content


async def test_unknown_skill_lists_the_available_ones() -> None:
    tools, _ = await build('alpha', 'beta')

    result = await tools.execute(
        ToolCallSegment(id='c1', name='skill', arguments='{"name": "gamma"}'),
    )

    assert "unknown skill 'gamma'" in result.content
    assert 'alpha, beta' in result.content
