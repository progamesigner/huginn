import json
from collections.abc import Mapping
from typing import override

from pydantic import JsonValue

from app.features.skills.sources.muninn import MuninnSkillSource
from lib.mcp import MCPClientContract, MCPToolInfo

_INCIDENT = """---
name: create-incident-report
description: Draft a structured incident report.
---
# Incident report

Follow these steps.
"""

_NO_DESC = """---
name: mystery
---
body only
"""

_NOTES = {
    'Agents/skills/create-incident-report/SKILL.md': _INCIDENT,
    'Agents/skills/mystery/SKILL.md': _NO_DESC,
}


class FakeMCP(MCPClientContract):
    calls: list[tuple[str, str, Mapping[str, JsonValue]]]

    def __init__(self) -> None:
        self.calls = []

    @override
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        self.calls.append((server, tool, dict(arguments)))
        if tool == 'list_memory_notes':
            return json.dumps({'items': list(_NOTES)})
        if tool == 'read_memory_note':
            return json.dumps({'content': _NOTES[str(arguments['path'])]})
        raise AssertionError(f'unexpected tool {tool!r}')

    @override
    async def list_tools(self, server: str) -> tuple[MCPToolInfo, ...]:
        return ()

    @override
    def unavailable(self) -> tuple[str, ...]:
        return ()


def source() -> tuple[MuninnSkillSource, FakeMCP]:
    client = FakeMCP()
    skills = MuninnSkillSource(
        client,
        'memory',
        'Agents/skills/**/SKILL.md',
    )
    return skills, client


async def test_list_parses_frontmatter_into_inventory() -> None:
    skills, client = source()

    inventory = await skills.list({'agent': 'default', 'user': 'swag'})

    assert {meta.name for meta in inventory} == {'create-incident-report', 'mystery'}
    incident = next(m for m in inventory if m.name == 'create-incident-report')
    assert incident.description == 'Draft a structured incident report.'
    assert incident.path == 'Agents/skills/create-incident-report/SKILL.md'
    assert client.calls[0][1] == 'list_memory_notes'
    assert client.calls[0][2] == {
        'glob': 'Agents/skills/**/SKILL.md',
        'agent': 'default',
        'user': 'swag',
    }


async def test_list_falls_back_to_name_when_description_missing() -> None:
    skills, _ = source()

    inventory = await skills.list({'agent': 'default', 'user': 'swag'})

    mystery = next(m for m in inventory if m.name == 'mystery')
    assert mystery.description == 'mystery'


async def test_load_returns_body_without_frontmatter() -> None:
    skills, _ = source()

    skill = await skills.load(
        'Agents/skills/create-incident-report/SKILL.md',
        {'agent': 'default', 'user': 'swag'},
    )

    assert skill.meta.name == 'create-incident-report'
    assert skill.body.startswith('# Incident report')
    assert 'name:' not in skill.body
