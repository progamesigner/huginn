from typing import Annotated, Literal

from pydantic import Field, HttpUrl, RootModel, SecretStr

from app.core.config import ConfigValue, HuginnConfig, SectionSpec

from .exceptions import UnknownToolError, UnknownToolKindError
from .registry import ToolRegistry


class WebSearchToolConfig(ConfigValue):
    kind: Annotated[Literal['websearch'], Field()]
    api_key: Annotated[SecretStr, Field()]
    base_url: Annotated[HttpUrl, Field()] = HttpUrl('https://openapi.felo.ai')


# NOTE: future tool kinds extend this union, discriminated by `kind`
type ToolConfig = WebSearchToolConfig


class ToolsConfig(RootModel[dict[str, ToolConfig]]):
    pass


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


def build_sections(registry: ToolRegistry) -> tuple[SectionSpec, ...]:
    return (
        SectionSpec(
            'tools',
            ToolsConfig,
            describe=describe_tools,
            validate=lambda config: validate_tools(config, registry),
        ),
    )
