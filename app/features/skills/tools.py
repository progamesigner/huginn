from collections.abc import Callable, Sequence
from typing import Annotated, override

from pydantic import Field

from app.features.agents.config import AgentConfig
from app.features.agents.contracts import NamedTool, ToolSource
from app.features.agents.types import ToolOutput, ToolParams, TypedTool

from .repositories import SkillRepository
from .types import SkillMeta


def _build_skill_tool(
    repository: SkillRepository,
    inventory: tuple[SkillMeta, ...],
) -> SkillTool:
    return SkillTool(repository, inventory)


async def _build_agent_skills(
    repository: SkillRepository,
    allow: Sequence[str],
) -> SkillTool:
    inventory = await repository.inventory()
    # NOTE: an empty allowlist offers every skill; a non-empty one narrows
    selected = tuple(meta for meta in inventory if not allow or meta.name in allow)
    return _build_skill_tool(repository, selected)


def _describe(inventory: tuple[SkillMeta, ...]) -> str:
    intro = (
        'Load specialized, step-by-step instructions for a task on demand.'
        ' Call this before a task that matches one of the available skills.'
    )
    if not inventory:
        return f'{intro} No skills are currently available.'

    entries = '\n'.join(
        f'  <skill name={meta.name!r}>{meta.description}</skill>' for meta in inventory
    )
    return f'{intro}\n<skills>\n{entries}\n</skills>'


class SkillParams(ToolParams):
    name: Annotated[str, Field(description='The name of the skill to load')]


class SkillTool(TypedTool[SkillParams]):
    name = 'skill'
    params = SkillParams

    repository: SkillRepository
    available: frozenset[str]

    def __init__(
        self,
        repository: SkillRepository,
        inventory: tuple[SkillMeta, ...],
    ) -> None:
        self.repository = repository
        self.available = frozenset(meta.name for meta in inventory)
        self.description = _describe(inventory)

    @override
    async def run(self, params: SkillParams) -> ToolOutput:
        skill = (
            await self.repository.load(params.name)
            if params.name in self.available
            else None
        )
        if skill is None:
            offered = ', '.join(sorted(self.available)) or 'none'
            return f'unknown skill {params.name!r}; available: {offered}'
        return skill.body


class SkillToolSource(ToolSource):
    repository: Callable[[], SkillRepository]

    def __init__(self, repository: Callable[[], SkillRepository]) -> None:
        self.repository = repository

    @override
    async def collect(self, agent: AgentConfig) -> Sequence[NamedTool]:
        if not agent.skills.enabled:
            return ()

        tool = await _build_agent_skills(self.repository(), agent.skills.allow)
        return ((tool.name, tool),)
