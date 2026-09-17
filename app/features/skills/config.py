from typing import Annotated

from pydantic import Field

from app.core.config.registry import HuginnConfig, SectionSpec
from app.features.agents.config import AGENTS, InjectionConfig

from .exceptions import SkillsNotConfiguredError

_DEFAULT_GLOB = 'Agents/skills/**/SKILL.md'


class SkillsSourceConfig(InjectionConfig):
    server: Annotated[str | None, Field()] = None
    glob: Annotated[str, Field()] = _DEFAULT_GLOB


def _describe_skills(config: HuginnConfig) -> list[str]:
    skills = config.section(SKILLS)
    if skills.server is None:
        return []
    return [f'skills: server={skills.server} glob={skills.glob}']


def _validate_agent_skill_refs(config: HuginnConfig) -> None:
    if config.section(SKILLS).server is not None:
        return

    for agent_name, agent in config.section(AGENTS).root.items():
        if agent.skills.enabled:
            raise SkillsNotConfiguredError(
                f'agents.{agent_name}.skills.enabled is set'
                f' but skills.server is not configured',
            )


SKILLS = SectionSpec(
    'skills',
    SkillsSourceConfig,
    describe=_describe_skills,
    validate=_validate_agent_skill_refs,
)
