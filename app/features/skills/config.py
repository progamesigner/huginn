from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

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
            raise ValueError('skills.source.glob must stay within its root')
        return self


type SkillSourceConfig = Annotated[
    MCPSkillSourceConfig | FilesystemSkillSourceConfig,
    Field(discriminator='kind'),
]


class SkillsConfig(ConfigValue):
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


def _describe_skills(config: HuginnConfig) -> list[str]:
    source = config.section(SKILLS).source
    if source is None:
        return []
    return [f'skills: kind={source.kind} glob={source.glob}']


def _validate_skill_refs(config: HuginnConfig) -> None:
    source = config.section(SKILLS).source
    if isinstance(source, MCPSkillSourceConfig):
        if source.server not in config.section(MCP_SERVERS).root:
            raise UnknownMCPServerReferenceError(
                f'skills.source.server: unknown mcp server {source.server!r}',
            )
    if source is not None:
        return
    for agent_name, agent in config.section(AGENTS).root.items():
        if agent.skills.enabled:
            raise SkillsNotConfiguredError(
                f'agents.{agent_name}.skills.enabled is set'
                f' but skills.source is not configured',
            )


SKILLS = SectionSpec(
    'skills',
    SkillsConfig,
    describe=_describe_skills,
    validate=_validate_skill_refs,
)
