import json
from collections.abc import Mapping
from typing import Any, override

from pydantic import JsonValue, TypeAdapter, ValidationError

from .config import ToolConfig
from .registry import ToolRegistry
from .types import Tool, ToolOutput


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


def _type_problem(annotation: Any, value: JsonValue) -> str | None:
    try:
        TypeAdapter(annotation).validate_python(value)
    except ValidationError as error:
        return '; '.join(item['msg'] for item in error.errors())

    return None


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


def check_overlay(
    registry: ToolRegistry,
    kind: str,
    params: Mapping[str, JsonValue],
    inject: Mapping[str, JsonValue],
) -> str | None:
    if shared := set(params) & set(inject):
        return f'params and inject cannot share keys: {", ".join(sorted(shared))}'

    model = registry.params_model(kind)
    fields = model.model_fields if model is not None else {}

    for source, overlay in (('params', params), ('inject', inject)):
        for key, value in overlay.items():
            if key not in fields:
                available = ', '.join(sorted(fields)) or 'none'
                return (
                    f'{source} references unknown parameter {key!r}; '
                    f'available: {available}'
                )

            problem = _type_problem(fields[key].rebuild_annotation(), value)
            if problem is not None:
                return f'{source}.{key}: {problem}'

    return None
