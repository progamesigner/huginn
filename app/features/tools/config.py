from collections.abc import Iterator
from functools import reduce
from operator import or_
from typing import Annotated, Any

from pydantic import BaseModel, Discriminator, Field, RootModel, Tag, create_model

from app.core.config import ConfigValue, HuginnConfig, SectionSpec

from .exceptions import UnknownToolError, UnknownToolKindError
from .registry import ToolRegistry


class ToolConfig(ConfigValue):
    kind: Annotated[str, Field()]


class ToolsConfig(RootModel[dict[str, ToolConfig]]):
    pass


def _tool_config_element(configs: dict[str, type[BaseModel] | None]) -> Any:
    # NOTE: the parse model is built from the registry
    if not configs:
        return ToolConfig

    members: Any = {kind: model or ToolConfig for kind, model in configs.items()}
    union = reduce(
        or_,
        (Annotated[model, Tag(kind)] for kind, model in members.items()),
    )
    return Annotated[union, Discriminator(_tool_kind)]


def _tool_kind(value: Any) -> str | None:
    return (
        value.get('kind') if isinstance(value, dict) else getattr(value, 'kind', None)
    )


def _build_tools_config(registry: ToolRegistry) -> type[BaseModel]:
    element = _tool_config_element(registry.config_models)
    return create_model(
        'ToolsConfig',
        __base__=ToolsConfig,
        root=(dict[str, element], ...),
    )


def describe_tools(config: HuginnConfig) -> list[str]:
    tools = config.section('tools', ToolsConfig).root
    entries = ', '.join(f'{name} ({tool.kind})' for name, tool in sorted(tools.items()))
    return [f'tools: {entries or "none"}']


def validate_tools(config: HuginnConfig, registry: ToolRegistry) -> None:
    for name, tool in config.section('tools', ToolsConfig).root.items():
        try:
            registry.get(tool.kind)
        except UnknownToolError as error:
            raise UnknownToolKindError(f'tools.{name}.kind: {error}') from error


def build_sections(registry: ToolRegistry) -> Iterator[SectionSpec]:
    yield SectionSpec(
        'tools',
        lambda: _build_tools_config(registry),
        describe=describe_tools,
        validate=lambda config: validate_tools(config, registry),
    )
