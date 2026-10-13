from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from app.core.config.exceptions import ConfigError
from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue
from app.features.agents.config import AGENTS, InjectionConfig
from app.features.mcp.config import MCP_SERVERS
from app.features.mcp.exceptions import UnknownMCPServerReferenceError

from .exceptions import SkillsNotConfiguredError


class MCPSkillSourceConfig(InjectionConfig):
    kind: Annotated[Literal['mcp'], Field()] = 'mcp'
    server: Annotated[str, Field(min_length=1)]
    list_tool: Annotated[str, Field(min_length=1)] = 'list_memory_notes'
    read_tool: Annotated[str, Field(min_length=1)] = 'read_memory_note'
    glob: Annotated[str, Field()] = 'Agents/skills/**/SKILL.md'


class FilesystemSkillSourceConfig(ConfigValue):
    kind: Annotated[Literal['filesystem'], Field()]
    root: Annotated[Path, Field()]
    glob: Annotated[str, Field()] = '**/SKILL.md'

    @model_validator(mode='after')
    def contained_glob(self) -> Self:
        if Path(self.glob).is_absolute() or '..' in Path(self.glob).parts:
            raise ValueError('skill source glob must stay within its root')
        return self


type SkillSourceConfig = Annotated[
    MCPSkillSourceConfig | FilesystemSkillSourceConfig,
    Field(discriminator='kind'),
]


class SkillsConfig(ConfigValue):
    sources: Annotated[dict[str, SkillSourceConfig], Field()] = Field(
        default_factory=dict,
    )
    source: Annotated[SkillSourceConfig | None, Field()] = None

    @model_validator(mode='before')
    @classmethod
    def require_source_shape(cls, value: object) -> object:
        if isinstance(value, dict) and value.keys() & {
            'server',
            'glob',
            'inject',
            'bind',
        }:
            raise ValueError(
                'move skill source fields under skills.source and set kind: mcp',
            )
        return value

    @model_validator(mode='after')
    def unambiguous_default(self) -> Self:
        if self.source is not None and 'default' in self.sources:
            raise ValueError(
                'skills.source and skills.sources.default cannot both be set;'
                ' move the legacy source to skills.sources.default',
            )
        if any(not name for name in self.sources):
            raise ValueError('skills.sources names must not be empty')
        return self

    @property
    def named_sources(self) -> dict[str, SkillSourceConfig]:
        return (
            {'default': self.source, **self.sources}
            if self.source is not None
            else self.sources
        )

    def resolve_source(self, name: str) -> SkillSourceConfig:
        if (source := self.named_sources.get(name)) is None:
            raise SkillsNotConfiguredError(
                f'skills.sources: source {name!r} is not configured',
            )
        return source


def _describe_skills(config: HuginnConfig) -> list[str]:
    return [
        f'skills.{name}: kind={source.kind} glob={source.glob}'
        for name, source in config.section(SKILLS).named_sources.items()
    ]


def _validate_skill_refs(config: HuginnConfig) -> None:
    sources = config.section(SKILLS).named_sources
    for name, source in sources.items():
        if isinstance(source, MCPSkillSourceConfig):
            if source.server not in config.section(MCP_SERVERS).root:
                raise UnknownMCPServerReferenceError(
                    f'skills.sources.{name}.server: unknown mcp server {source.server!r}',
                )
    for agent_name, agent in config.section(AGENTS).root.items():
        name = agent.skills.source or 'default'
        if name in sources:
            continue
        if agent.skills.source is not None:
            raise ConfigError(
                f'agents.{agent_name}.skills.source: unknown skill source {name!r};'
                ' configure it in skills.sources',
            )
        if agent.skills.enabled:
            raise SkillsNotConfiguredError(
                f'agents.{agent_name}.skills.enabled is set but skill source'
                f' {name!r} is not configured; set skills.sources.default'
                ' or select a named source with skills.source on the agent',
            )


SKILLS = SectionSpec(
    'skills',
    SkillsConfig,
    describe=_describe_skills,
    validate=_validate_skill_refs,
)
