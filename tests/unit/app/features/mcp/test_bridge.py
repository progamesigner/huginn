from collections.abc import Mapping
from typing import override

from pydantic import JsonValue

from app.features.mcp.bridge import MCPTool, build_mcp_tools, namespaced
from lib.mcp import MCPClientContract, MCPToolInfo


class FakeClient(MCPClientContract):
    calls: list[tuple[str, str, dict[str, JsonValue]]]

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
        return 'called'

    @override
    async def list_tools(self, server: str) -> tuple[MCPToolInfo, ...]:
        return ()

    @override
    def unavailable(self) -> tuple[str, ...]:
        return ()


def _info(name: str, schema: JsonValue | None = None) -> MCPToolInfo:
    return MCPToolInfo(
        name=name,
        description=f'{name} description',
        input_schema=schema or {'type': 'object', 'properties': {}, 'required': []},
    )


def test_namespaced_prefixes_server_and_tool() -> None:
    assert namespaced('memory', 'recall') == 'mcp__memory__recall'


def test_build_denies_every_tool_when_allow_is_empty() -> None:
    tools = build_mcp_tools(FakeClient(), 'memory', [_info('a'), _info('b')], (), {})

    assert tools == []


def test_build_filters_to_the_allowlist() -> None:
    tools = build_mcp_tools(
        FakeClient(),
        'memory',
        [_info('a'), _info('b')],
        ('a',),
        {},
    )

    assert [tool.name for tool in tools] == ['mcp__memory__a']


def test_schema_hides_injected_parameters() -> None:
    schema = {
        'type': 'object',
        'properties': {'query': {'type': 'string'}, 'user': {'type': 'string'}},
        'required': ['query', 'user'],
    }
    tool = MCPTool(FakeClient(), 'memory', _info('recall', schema), {'user': 'swag'})

    result = tool.schema()

    assert isinstance(result, dict)
    assert result['properties'] == {'query': {'type': 'string'}}
    assert result['required'] == ['query']


async def test_invoke_merges_injected_parameters() -> None:
    client = FakeClient()
    tool = MCPTool(
        client,
        'memory',
        _info('recall'),
        {'agent': 'default', 'user': 'swag'},
    )

    output = await tool.invoke('{"query": "hi"}')

    assert output == 'called'
    assert client.calls == [
        ('memory', 'recall', {'query': 'hi', 'agent': 'default', 'user': 'swag'}),
    ]


async def test_invoke_tolerates_empty_arguments() -> None:
    client = FakeClient()
    tool = MCPTool(client, 'memory', _info('ping'), {})

    await tool.invoke('')

    assert client.calls == [('memory', 'ping', {})]


def test_delegation_is_declared_per_server_rather_than_guessed_from_names() -> None:
    tools = build_mcp_tools(
        FakeClient(),
        'tasks',
        [_info('task'), _info('dispatch')],
        ('task', 'dispatch'),
        {},
        ('dispatch',),
    )

    assert {tool.tool: tool.execution_policy for tool in tools} == {
        'task': 'ordinary',
        'dispatch': 'delegation',
    }


def test_delegate_tools_must_be_allowed() -> None:
    import pytest
    from pydantic import ValidationError

    from app.features.agents.config import MCPRef

    with pytest.raises(ValidationError, match='delegate'):
        MCPRef(server='tasks', allow=('task',), delegate=('dispatch',))


async def test_only_the_root_brief_is_protected_from_generic_mutations() -> None:
    client = FakeClient()
    write = MCPTool(client, 'memory', _info('write_memory_note'), {})
    rename = MCPTool(client, 'memory', _info('rename_memory_note'), {})

    for path in ('Agents/projects/x/AGENTS.md', 'Agents', 'notes/AGENTS.md'):
        assert await write.invoke(f'{{"path": "{path}", "content": "x"}}') == 'called'
    for path in ('Agents/AGENTS.md', './Agents//AGENTS.md', '/agents/agents.md'):
        assert 'managed' in str(await write.invoke(f'{{"path": "{path}"}}'))
    assert 'managed' in str(await rename.invoke('{"from": "Agents", "to": "Old"}'))
    assert len(client.calls) == 3


async def test_brief_properties_are_managed_but_replacement_text_is_content() -> None:
    client = FakeClient()
    properties = MCPTool(client, 'memory', _info('update_note_properties'), {})
    edit = MCPTool(client, 'memory', _info('edit_memory_note'), {})

    assert 'managed' in str(
        await properties.invoke('{"path": "Agents/AGENTS.md", "properties": {}}'),
    )
    assert (
        await edit.invoke(
            '{"path": "Agents/notes.md", "search_string": "x",'
            ' "replace_string": "Agents/AGENTS.md"}',
        )
        == 'called'
    )
