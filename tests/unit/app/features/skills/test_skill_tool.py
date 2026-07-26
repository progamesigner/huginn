from datetime import timedelta
from typing import override

from app.features.agents.config import AgentConfig, SkillsConfig
from app.features.agents.types import ToolSet
from app.features.skills.repositories import SkillRepository
from app.features.skills.sources.contracts import SkillSource
from app.features.skills.tools import SkillParams, SkillToolSource, _build_skill_tool
from app.features.skills.types import Skill, SkillMeta
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
    tools.add('skill', _build_skill_tool(repository, inventory))
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


def _source(*names: str) -> SkillToolSource:
    source = FakeSource(tuple(_meta(name) for name in names))
    repository = SkillRepository(source, FakeClock(), ttl=timedelta(minutes=15))
    return SkillToolSource(lambda: repository)


def _agent(*, enabled: bool = True, allow: tuple[str, ...] = ()) -> AgentConfig:
    return AgentConfig(
        description='x',
        instructions='y',
        skills=SkillsConfig(enabled=enabled, allow=allow),
    )


async def test_source_narrows_the_inventory_to_the_allowlist() -> None:
    named = await _source('triage', 'report').collect(_agent(allow=('triage',)))

    assert 'triage' in named[0][1].description
    assert 'report' not in named[0][1].description


async def test_source_offers_everything_when_the_allowlist_is_empty() -> None:
    named = await _source('triage', 'report').collect(_agent())

    assert 'triage' in named[0][1].description
    assert 'report' in named[0][1].description


async def test_source_names_the_tool_so_agents_need_not_hardcode_it() -> None:
    named = await _source('triage').collect(_agent())

    assert [name for name, _ in named] == ['skill']


async def test_source_contributes_nothing_when_the_agent_disables_skills() -> None:
    named = await _source('triage').collect(_agent(enabled=False))

    assert named == ()


async def test_unknown_skill_lists_the_available_ones() -> None:
    tools, _ = await build('alpha', 'beta')

    result = await tools.execute(
        ToolCallSegment(id='c1', name='skill', arguments='{"name": "gamma"}'),
    )

    assert "unknown skill 'gamma'" in result.content
    assert 'alpha, beta' in result.content
