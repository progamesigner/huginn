from typing import Annotated

from pydantic import Field

from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue

_DEFAULT_GLOB = 'Agents/skills/**/SKILL.md'


class SkillsSourceConfig(ConfigValue):
    server: Annotated[str | None, Field()] = None
    inject: Annotated[dict[str, str], Field()] = Field(default_factory=dict)
    glob: Annotated[str, Field()] = _DEFAULT_GLOB


def _describe_skills(config: HuginnConfig) -> list[str]:
    skills = config.section(SKILLS)
    if skills.server is None:
        return []
    return [f'skills: server={skills.server} glob={skills.glob}']


SKILLS = SectionSpec('skills', SkillsSourceConfig, describe=_describe_skills)
