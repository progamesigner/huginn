from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, override

from pydantic import Field, JsonValue

from app.features.agents.config import AgentConfig
from app.features.agents.contracts import (
    ToolHarvest,
    ToolOutput,
    ToolParams,
    ToolSource,
    TypedTool,
)
from app.features.agents.entities import TurnScope, resolve_bindings
from lib.observability import Logger

from .config import SkillsSourceConfig
from .entities import SkillMeta
from .repositories import SkillRepository

_LIBRARY = 'skills'


def _build_skill_tool(
    repository: SkillRepository,
    inventory: tuple[SkillMeta, ...],
    scope: Mapping[str, JsonValue],
) -> SkillTool:
    return SkillTool(repository, inventory, scope)


async def _build_agent_skills(
    repository: SkillRepository,
    allow: Sequence[str],
    scope: Mapping[str, JsonValue],
) -> SkillTool:
    inventory = await repository.inventory(scope)
    # NOTE: an empty allowlist offers every skill; a non-empty one narrows
    selected = tuple(meta for meta in inventory if not allow or meta.name in allow)
    return _build_skill_tool(repository, selected, scope)


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
    scope: dict[str, JsonValue]

    def __init__(
        self,
        repository: SkillRepository,
        inventory: tuple[SkillMeta, ...],
        scope: Mapping[str, JsonValue],
    ) -> None:
        self.repository = repository
        self.available = frozenset(meta.name for meta in inventory)
        self.scope = dict(scope)
        self.description = _describe(inventory)

    @override
    async def run(self, params: SkillParams) -> ToolOutput:
        skill = (
            await self.repository.load(params.name, self.scope)
            if params.name in self.available
            else None
        )
        if skill is None:
            offered = ', '.join(sorted(self.available)) or 'none'
            return f'unknown skill {params.name!r}; available: {offered}'
        return skill.body


class SkillToolSource(ToolSource):
    logger: Logger
    config: SkillsSourceConfig
    repository: Callable[[], SkillRepository]

    def __init__(
        self,
        repository: Callable[[], SkillRepository],
        config: SkillsSourceConfig,
        *,
        logger: Logger,
    ) -> None:
        self.logger = logger.bind(component='skills')
        self.config = config
        self.repository = repository

    @override
    async def collect(self, agent: AgentConfig, scope: TurnScope) -> ToolHarvest:
        if not agent.skills.enabled:
            return ToolHarvest()

        inject = resolve_bindings(self.config.inject, self.config.bind, scope)
        if inject is None:
            self.logger.info(
                'skill library omitted because bound keys are unavailable: %s',
                ', '.join(sorted(self.config.bind)),
            )
            return ToolHarvest()

        try:
            tool = await _build_agent_skills(
                self.repository(),
                agent.skills.allow,
                inject,
            )
        except Exception:
            self.logger.exception(
                'Could not read the skill inventory; offering no skills',
            )
            return ToolHarvest(unreachable=(_LIBRARY,))

        return ToolHarvest(tools=((tool.name, tool),))
