from collections.abc import Iterator, Mapping
from dataclasses import replace
from functools import reduce
from operator import or_
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    JsonValue,
    RootModel,
    Tag,
    TypeAdapter,
    ValidationError,
    create_model,
)

from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue
from app.features.agents.config import AGENTS

from .exceptions import (
    InvalidToolOverlayError,
    UnknownToolError,
    UnknownToolKindError,
    UnknownToolReferenceError,
)
from .registry import ToolRegistry


class ToolConfig(ConfigValue):
    kind: Annotated[str, Field()]
    description: Annotated[str | None, Field()] = None


class ToolConfigList(RootModel[dict[str, ToolConfig]]):
    pass


def _build_tools_config(registry: ToolRegistry) -> type[BaseModel]:
    element = _tool_config_element(registry.config_models)
    return create_model(
        'ToolConfigList',
        __base__=ToolConfigList,
        root=(dict[str, element], ...),
    )


def _check_overlay(
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


def _describe_tools(config: HuginnConfig) -> list[str]:
    tools = config.section(TOOLS).root
    entries = ', '.join(f'{name} ({tool.kind})' for name, tool in sorted(tools.items()))
    return [f'tools: {entries or "none"}']


def _tagged(kind: str, model: Any) -> Any:
    return Annotated[model, Tag(kind)]


def _tool_config_element(configs: dict[str, type[BaseModel] | None]) -> Any:
    # NOTE: the parse model is built from the registry
    if not configs:
        return ToolConfig

    union = reduce(
        or_,
        (_tagged(kind, model or ToolConfig) for kind, model in configs.items()),
    )
    return Annotated[union, Discriminator(_tool_kind)]


def _tool_kind(value: object) -> str | None:
    match value:
        case {'kind': str(kind)} | object(kind=str(kind)):
            return kind
        case _:
            return None


def _type_problem(annotation: Any, value: JsonValue) -> str | None:
    try:
        TypeAdapter(annotation).validate_python(value)
    except ValidationError as error:
        return '; '.join(item['msg'] for item in error.errors())

    return None


def _validate_agent_tool_refs(config: HuginnConfig, registry: ToolRegistry) -> None:
    configured = config.section(TOOLS).root

    for agent_name, agent in config.section(AGENTS).root.items():
        for index, ref in enumerate(agent.tools):
            base = f'agents.{agent_name}.tools[{index}]'
            if ref.name not in configured:
                available = ', '.join(sorted(configured)) or 'none'
                raise UnknownToolReferenceError(
                    f'{base}.name: unknown tool {ref.name!r}; available: {available}',
                )

            problem = _check_overlay(
                registry,
                configured[ref.name].kind,
                ref.params,
                ref.inject,
            )
            if problem is not None:
                raise InvalidToolOverlayError(f'{base}: {problem}')


def _validate_tools(config: HuginnConfig, registry: ToolRegistry) -> None:
    for name, tool in config.section(TOOLS).root.items():
        try:
            registry.get(tool.kind)
        except UnknownToolError as error:
            raise UnknownToolKindError(f'tools.{name}.kind: {error}') from error


def _validate_tool_section(config: HuginnConfig, registry: ToolRegistry) -> None:
    _validate_tools(config, registry)
    _validate_agent_tool_refs(config, registry)


def build_sections(registry: ToolRegistry) -> Iterator[SectionSpec[BaseModel]]:
    # NOTE: the thunk and validator need the runtime registry using replace()
    yield replace(
        TOOLS,
        thunk=lambda: _build_tools_config(registry),
        validate=lambda config: _validate_tool_section(config, registry),
    )


TOOLS = SectionSpec(
    'tools',
    ToolConfigList,
    describe=_describe_tools,
)
