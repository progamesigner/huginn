from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator

from app.core.config.exceptions import ConfigError
from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue
from app.features.agents.config import AGENTS, InjectionConfig
from app.features.agents.entities import TurnScope, resolve_bindings
from app.features.mcp.config import MCP_SERVERS

from .entities import AssetKind


class AssetGlobs(ConfigValue):
    skill: Annotated[str, Field(min_length=1)] = 'Agents/skills/**/SKILL.md'
    command: Annotated[str, Field(min_length=1)] = 'Agents/commands/*.md'
    subagent: Annotated[str, Field(min_length=1)] = 'Agents/agents/*/PROMPT.md'

    @model_validator(mode='after')
    def contained(self) -> Self:
        for value in (self.skill, self.command, self.subagent):
            if Path(value).is_absolute() or '..' in Path(value).parts:
                raise ValueError('asset glob must stay within its root')
        return self

    def for_kind(self, kind: AssetKind) -> str:
        return getattr(self, kind)


class MCPAssetSourceConfig(InjectionConfig):
    kind: Annotated[Literal['mcp'], Field()] = 'mcp'
    server: Annotated[str, Field(min_length=1)]
    list_tool: Annotated[str, Field(min_length=1)] = 'list_memory_notes'
    read_tool: Annotated[str, Field(min_length=1)] = 'read_memory_note'
    globs: Annotated[AssetGlobs, Field()] = Field(default_factory=AssetGlobs)


class FilesystemAssetSourceConfig(ConfigValue):
    kind: Annotated[Literal['filesystem'], Field()]
    root: Annotated[Path, Field()]
    globs: Annotated[AssetGlobs, Field()] = Field(default_factory=AssetGlobs)


type AssetSourceConfig = Annotated[
    MCPAssetSourceConfig | FilesystemAssetSourceConfig,
    Field(discriminator='kind'),
]


class AssetsConfig(ConfigValue):
    sources: Annotated[dict[str, AssetSourceConfig], Field()] = Field(
        default_factory=dict,
    )
    max_entries: Annotated[int, Field(gt=0)] = 256

    @model_validator(mode='after')
    def named(self) -> Self:
        if any(not name.strip() for name in self.sources):
            raise ValueError('asset source names must not be empty')
        return self

    def resolve_source(self, name: str) -> AssetSourceConfig:
        if name not in self.sources:
            raise ConfigError(f'assets.sources: unknown source {name!r}')
        return self.sources[name]

    def binding(self, name: str, scope: TurnScope) -> dict[str, JsonValue] | None:
        source = self.resolve_source(name)
        return (
            resolve_bindings(source.inject, source.bind, scope)
            if isinstance(source, MCPAssetSourceConfig)
            else {}
        )


def _validate(config: HuginnConfig) -> None:
    assets = config.section(ASSETS)
    for name, source in assets.sources.items():
        if isinstance(source, MCPAssetSourceConfig):
            if source.server not in config.section(MCP_SERVERS).root:
                raise ConfigError(
                    f'assets.sources.{name}: unknown server {source.server!r}',
                )
    for name, agent in config.section(AGENTS).root.items():
        for kind in ('commands', 'subagents', 'brief'):
            setting = getattr(agent, kind, None)
            if setting is not None and (setting.enabled or setting.source is not None):
                if (
                    kind == 'subagents'
                    and setting.definitions
                    and setting.source is None
                    and 'default' not in assets.sources
                ):
                    continue
                source = assets.resolve_source(setting.source or 'default')
                if (
                    kind == 'brief'
                    and setting.writable
                    and not isinstance(source, MCPAssetSourceConfig)
                ):
                    raise ConfigError(
                        f'agents.{name}.brief: writable requires an MCP source',
                    )


ASSETS = SectionSpec('assets', AssetsConfig, validate=_validate)


def asset_settings(config: HuginnConfig) -> AssetsConfig:
    # A directly supplied config may omit optional sections, just as YAML may.
    return config.section(ASSETS) if ASSETS.key in config.sections else AssetsConfig()
