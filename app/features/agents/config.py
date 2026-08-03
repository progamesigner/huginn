from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, JsonValue, RootModel

from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue

from .exceptions import AgentError, AmbiguousInstructionsError, MissingModelError


class DefaultsConfig(ConfigValue):
    model: Annotated[str | None, Field()] = None
    router_prompt: Annotated[str | None, Field()] = None


class ToolRef(ConfigValue):
    name: Annotated[str, Field()]
    params: Annotated[dict[str, JsonValue], Field()] = Field(default_factory=dict)
    inject: Annotated[dict[str, JsonValue], Field()] = Field(default_factory=dict)


class MCPRef(ConfigValue):
    server: Annotated[str, Field()]
    allow: Annotated[tuple[str, ...], Field()] = ()
    inject: Annotated[dict[str, str], Field()] = Field(default_factory=dict)


class SkillsConfig(ConfigValue):
    enabled: Annotated[bool, Field()] = False
    allow: Annotated[tuple[str, ...], Field()] = ()


class AgentConfig(ConfigValue):
    description: Annotated[str, Field()]
    model: Annotated[str | None, Field()] = None
    instructions: Annotated[str | None, Field()] = None
    instructions_file: Annotated[str | None, Field()] = None
    tools: Annotated[tuple[ToolRef, ...], Field()] = ()
    mcp: Annotated[tuple[MCPRef, ...], Field()] = ()
    skills: Annotated[SkillsConfig, Field()] = Field(default_factory=SkillsConfig)


class AgentConfigList(RootModel[dict[str, AgentConfig]]):
    pass


def _describe_agent(name: str, agent: AgentConfig) -> str:
    model = agent.model or 'default'
    tools = ', '.join(tool.name for tool in agent.tools) or 'none'
    servers = ', '.join(ref.server for ref in agent.mcp) or 'none'
    return f'  {name}: model={model} tools={tools} mcp={servers}'


def _describe_agents(config: HuginnConfig) -> list[str]:
    # NOTE: instructions are secrets by policy; the summary must never echo them
    agents = config.section(AGENTS).root
    lines = [f'agents ({len(agents)}):']
    lines.extend(_describe_agent(name, agent) for name, agent in agents.items())
    return lines


def _validate_agents(config: HuginnConfig) -> None:
    default_model = config.section(DEFAULTS).model

    for agent_name, agent in config.section(AGENTS).root.items():
        base = f'agents.{agent_name}'

        if (agent.instructions is None) == (agent.instructions_file is None):
            raise AmbiguousInstructionsError(
                f'{base}: exactly one of instructions/instructions_file is required',
            )

        try:
            resolve_model(agent.model, default_model)
        except MissingModelError as error:
            raise MissingModelError(f'{base}.model: {error}') from error


def build_sections() -> Iterator[SectionSpec[BaseModel]]:
    yield replace(AGENTS, validate=_validate_agents)
    yield DEFAULTS


def resolve_model(explicit: str | None, default: str | None) -> str:
    if (model := explicit or default) is not None:
        return model

    raise MissingModelError('no model configured; set the model or defaults.model')


def load_instructions(agent: AgentConfig) -> str:
    if agent.instructions is not None:
        return agent.instructions

    if agent.instructions_file is not None:
        return Path(agent.instructions_file).read_text(encoding='utf-8')

    raise AgentError('agent has neither instructions nor instructions_file')


AGENTS = SectionSpec(
    'agents',
    AgentConfigList,
    required=True,
    describe=_describe_agents,
)

DEFAULTS = SectionSpec('defaults', DefaultsConfig)
