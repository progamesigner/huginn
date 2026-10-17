import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import override

from pydantic import JsonValue

from app.features.agents.contracts import Tool, ToolOutput
from lib.mcp import MCPClientContract, MCPToolInfo


def _without(schema: JsonValue, keys: Iterable[str]) -> JsonValue:
    if not isinstance(schema, dict):
        return schema

    hidden = set(keys)
    result = dict(schema)

    properties = result.get('properties')
    if isinstance(properties, dict):
        result['properties'] = {
            key: value for key, value in properties.items() if key not in hidden
        }

    required = result.get('required')
    if isinstance(required, list):
        result['required'] = [item for item in required if item not in hidden]

    return result


class MCPTool(Tool):
    client: MCPClientContract
    server: str
    tool: str
    input_schema: JsonValue
    inject: dict[str, JsonValue]

    def __init__(
        self,
        client: MCPClientContract,
        server: str,
        info: MCPToolInfo,
        inject: Mapping[str, JsonValue],
    ) -> None:
        self.name = namespaced(server, info.name)
        self.description = info.description
        self.client = client
        self.server = server
        self.tool = info.name
        if info.name in {
            'agent',
            'spawn_agent',
            'delegate',
            'run_agent',
            'call_agent',
            'task',
            'create_task',
        }:
            self.execution_policy = 'delegation'
        self.writes_brief = info.name in {'update_agents_file', 'update_brief'}
        self.input_schema = info.input_schema
        self.inject = dict(inject)

    @override
    async def invoke(self, arguments: str) -> ToolOutput:
        provided = json.loads(arguments) if arguments.strip() else {}
        payload = {**provided, **self.inject}
        if self.writes_brief:
            return 'brief updates require the managed update_brief tool and an explicit user request'
        if self.tool in _MUTATIONS and _targets_brief(payload):
            return (
                'AGENTS.md is managed; use update_brief with an explicit user request'
            )
        return await self.client.call_tool(self.server, self.tool, payload)

    @override
    def schema(self) -> JsonValue:
        # NOTE: injected parameters are supplied by the harness
        return _without(self.input_schema, self.inject)


def build_mcp_tools(
    client: MCPClientContract,
    server: str,
    infos: Sequence[MCPToolInfo],
    allow: Sequence[str],
    inject: Mapping[str, JsonValue],
) -> list[MCPTool]:
    # NOTE: deny by default
    return [
        MCPTool(client, server, info, inject) for info in infos if info.name in allow
    ]


def namespaced(server: str, tool: str) -> str:
    return f'mcp__{server}__{tool}'


_MUTATIONS = frozenset(
    {
        'write_memory_note',
        'write_memory_notes',
        'edit_memory_note',
        'edit_memory_notes',
        'delete_memory_note',
        'delete_memory_notes',
        'rename_memory_note',
        'rename_memory_notes',
        'evolve_core_persona',
    },
)


def _targets_brief(value: JsonValue) -> bool:
    if isinstance(value, dict):
        return any(_targets_brief(item) for item in value.values())
    if isinstance(value, list):
        return any(_targets_brief(item) for item in value)
    if isinstance(value, str):
        path = value.replace('\\', '/').rstrip('/')
        return (
            path == 'Agents'
            or re.search(r'(?:^|/)AGENTS(?:[.][^/]*)?[.]md$', path, re.IGNORECASE)
            is not None
        )
    return False
