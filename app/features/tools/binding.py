import json
from collections.abc import Mapping
from typing import override

from pydantic import JsonValue

from app.features.agents.types import Tool, ToolOutput

from .config import ToolConfig
from .registry import ToolRegistry


def _apply_overlay(
    schema: JsonValue,
    params: Mapping[str, JsonValue],
    inject: Mapping[str, JsonValue],
) -> JsonValue:
    if not isinstance(schema, dict):
        return schema

    result = dict(schema)

    properties = result.get('properties')
    if isinstance(properties, dict):
        result['properties'] = {
            key: _property(value, params[key]) if key in params else value
            for key, value in properties.items()
            if key not in inject
        }

    required = result.get('required')
    if isinstance(required, list):
        overridden = {*inject, *params}
        result['required'] = [item for item in required if item not in overridden]

    return result


def _property(schema: JsonValue, default: JsonValue) -> JsonValue:
    if isinstance(schema, dict):
        return {**schema, 'default': default}

    return schema


class ConfiguredTool(Tool):
    tool: Tool
    params: dict[str, JsonValue]
    inject: dict[str, JsonValue]

    def __init__(
        self,
        tool: Tool,
        params: Mapping[str, JsonValue],
        inject: Mapping[str, JsonValue],
        description: str,
    ) -> None:
        self.name = tool.name
        self.description = description
        self.tool = tool
        self.params = dict(params)
        self.inject = dict(inject)

    @override
    async def invoke(self, arguments: str) -> ToolOutput:
        provided = json.loads(arguments) if arguments.strip() else {}
        # NOTE: inject overrides the model, which overrides per-agent defaults
        merged = {**self.params, **provided, **self.inject}
        return await self.tool.invoke(json.dumps(merged))

    @override
    def schema(self) -> JsonValue:
        return _apply_overlay(self.tool.schema(), self.params, self.inject)


def build_bound_tool(
    registry: ToolRegistry,
    config: ToolConfig,
    params: Mapping[str, JsonValue],
    inject: Mapping[str, JsonValue],
) -> Tool:
    base = registry.build(config.kind, config)
    return ConfiguredTool(base, params, inject, config.description or base.description)
