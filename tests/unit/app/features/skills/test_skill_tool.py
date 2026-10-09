import builtins
from collections.abc import Mapping
from datetime import timedelta
from typing import override

import pytest
from pydantic import JsonValue

from app.features.agents.config import AgentConfig, SkillsConfig
from app.features.agents.entities import TurnScope
from app.features.agents.tooling import ToolSet
from app.features.skills.config import MCPSkillSourceConfig
from app.features.skills.config import SkillsConfig as SourceConfig
from app.features.skills.entities import Skill, SkillMeta
from app.features.skills.repositories import SkillRepository
from app.features.skills.sources.contracts import SkillSource
from app.features.skills.tools import SkillParams, SkillToolSource, _build_skill_tool
from lib.llm import ToolCallSegment
from tests.helpers.clock import FakeClock
from tests.helpers.logging import make_logger


class FakeSource(SkillSource):
    metas: tuple[SkillMeta, ...]
    list_scopes: builtins.list[dict[str, JsonValue]]
    load_scopes: builtins.list[dict[str, JsonValue]]

    def __init__(self, metas: tuple[SkillMeta, ...]) -> None:
        self.metas = metas
        self.list_scopes = []
        self.load_scopes = []

    @override
    async def list(self, scope: Mapping[str, JsonValue]) -> tuple[SkillMeta, ...]:
        self.list_scopes.append(dict(scope))
        return self.metas

    @override
    async def load(self, path: str, scope: Mapping[str, JsonValue]) -> Skill:
        self.load_scopes.append(dict(scope))
        meta = next(meta for meta in self.metas if meta.path == path)
        return Skill(meta=meta, body=f'# {meta.name}\n\ndo the thing')


def _meta(name: str) -> SkillMeta:
    return SkillMeta(name=name, description=f'{name} does things', path=f's/{name}.md')


async def build(*names: str) -> tuple[ToolSet, SkillRepository]:
    source = FakeSource(tuple(_meta(name) for name in names))
    repository = SkillRepository(source, FakeClock(), ttl=timedelta(minutes=15))
    inventory = await repository.inventory({})
    tools = ToolSet(logger=make_logger())
    tools.add('skill', _build_skill_tool(repository, inventory, {}))
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

    result, _ = await tools.execute(
        ToolCallSegment(id='c1', name='skill', arguments='{"name": "alpha"}'),
    )

    assert not result.error
    assert 'do the thing' in result.content


class BrokenSource(SkillSource):
    @override
    async def list(self, scope: Mapping[str, JsonValue]) -> tuple[SkillMeta, ...]:
        raise RuntimeError('memory: not connected')

    @override
    async def load(self, path: str, scope: Mapping[str, JsonValue]) -> Skill:
        raise RuntimeError('memory: not connected')


def _broken_repository() -> SkillRepository:
    return SkillRepository(BrokenSource(), FakeClock(), ttl=timedelta(minutes=15))


def _source(*names: str) -> SkillToolSource:
    source = FakeSource(tuple(_meta(name) for name in names))
    repository = SkillRepository(source, FakeClock(), ttl=timedelta(minutes=15))
    return SkillToolSource(
        lambda: repository,
        SourceConfig(),
        logger=make_logger(),
    )


def _agent(*, enabled: bool = True, allow: tuple[str, ...] = ()) -> AgentConfig:
    return AgentConfig(
        description='x',
        instructions='y',
        skills=SkillsConfig(enabled=enabled, allow=allow),
    )


_SCOPE = TurnScope(actor='user:1', agent='test', channel='test', chat='1')


async def test_source_narrows_the_inventory_to_the_allowlist() -> None:
    harvest = await _source('triage', 'report').collect(
        _agent(allow=('triage',)),
        _SCOPE,
    )

    assert 'triage' in harvest.tools[0][1].description
    assert 'report' not in harvest.tools[0][1].description


async def test_source_offers_everything_when_the_allowlist_is_empty() -> None:
    harvest = await _source('triage', 'report').collect(_agent(), _SCOPE)

    assert 'triage' in harvest.tools[0][1].description
    assert 'report' in harvest.tools[0][1].description


async def test_source_names_the_tool_so_agents_need_not_hardcode_it() -> None:
    harvest = await _source('triage').collect(_agent(), _SCOPE)

    assert [name for name, _ in harvest.tools] == ['skill']


async def test_source_contributes_nothing_when_the_agent_disables_skills() -> None:
    harvest = await _source('triage').collect(_agent(enabled=False), _SCOPE)

    assert harvest.tools == ()
    assert harvest.unreachable == ()


async def test_source_withholds_skills_when_a_bound_scope_is_missing() -> None:
    source = FakeSource((_meta('triage'),))
    repository = SkillRepository(source, FakeClock(), ttl=timedelta(minutes=15))
    tools = SkillToolSource(
        lambda: repository,
        SourceConfig(
            source=MCPSkillSourceConfig(server='memory', bind={'user': 'handle'}),
        ),
        logger=make_logger(),
    )

    harvest = await tools.collect(_agent(), _SCOPE)

    assert harvest.tools == ()


async def test_source_carries_bound_scope_from_inventory_into_load() -> None:
    source = FakeSource((_meta('triage'),))
    repository = SkillRepository(source, FakeClock(), ttl=timedelta(minutes=15))
    tools = SkillToolSource(
        lambda: repository,
        SourceConfig(
            source=MCPSkillSourceConfig(
                server='memory',
                inject={'agent': 'default'},
                bind={'user': 'handle'},
            ),
        ),
        logger=make_logger(),
    )
    scope = TurnScope(
        actor='user:1',
        agent='test',
        channel='test',
        chat='1',
        handle='alice',
    )

    harvest = await tools.collect(_agent(), scope)
    result = await harvest.tools[0][1].invoke('{"name":"triage"}')

    assert isinstance(result, str)
    assert 'do the thing' in result
    assert source.list_scopes == [{'agent': 'default', 'user': 'alice'}]
    assert source.load_scopes == [{'agent': 'default', 'user': 'alice'}]


# NOTE: the inventory is read off a remote server, so this is a thing that happens --
# raising took the whole turn down over a library the agent could have answered without
async def test_a_source_that_cannot_read_the_inventory_offers_no_skills(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = SkillToolSource(
        lambda: _broken_repository(),
        SourceConfig(),
        logger=make_logger(),
    )

    with caplog.at_level('ERROR'):
        harvest = await source.collect(_agent(), _SCOPE)

    assert harvest.tools == ()
    assert harvest.unreachable == ('skills',)
    assert any('skill inventory' in message for message in caplog.messages)


async def test_unknown_skill_lists_the_available_ones() -> None:
    tools, _ = await build('alpha', 'beta')

    result, _ = await tools.execute(
        ToolCallSegment(id='c1', name='skill', arguments='{"name": "gamma"}'),
    )

    assert "unknown skill 'gamma'" in result.content
    assert 'alpha, beta' in result.content
