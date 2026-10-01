import json
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
        self.input_schema = info.input_schema
        self.inject = dict(inject)

    @override
    async def invoke(self, arguments: str) -> ToolOutput:
        provided = json.loads(arguments) if arguments.strip() else {}
        return await self.client.call_tool(
            self.server,
            self.tool,
            {**provided, **self.inject},
        )

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
