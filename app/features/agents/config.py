from collections.abc import Iterator
from dataclasses import replace
from typing import Annotated

from pydantic import BaseModel, Field, JsonValue, RootModel

from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue
from app.features.tools.binding import check_overlay
from app.features.tools.config import TOOLS
from app.features.tools.mcp.config import MCP_SERVERS
from app.features.tools.registry import ToolRegistry

from .exceptions import (
    AmbiguousInstructionsError,
    InvalidToolOverlayError,
    MissingModelError,
    UnknownMCPServerReferenceError,
    UnknownToolReferenceError,
)


class DefaultsConfig(ConfigValue):
    model: Annotated[str | None, Field()] = None


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


def _validate_agents(config: HuginnConfig, registry: ToolRegistry) -> None:
    mcp_servers = config.section(MCP_SERVERS).root
    tools = config.section(TOOLS).root
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

        for index, tool in enumerate(agent.tools):
            if tool.name not in tools:
                available = ', '.join(sorted(tools)) or 'none'
                raise UnknownToolReferenceError(
                    f'{base}.tools[{index}].name: unknown tool {tool.name!r}; '
                    f'available: {available}',
                )

            problem = check_overlay(
                registry,
                tools[tool.name].kind,
                tool.params,
                tool.inject,
            )
            if problem is not None:
                raise InvalidToolOverlayError(f'{base}.tools[{index}]: {problem}')

        for index, mcp in enumerate(agent.mcp):
            if mcp.server not in mcp_servers:
                available = ', '.join(sorted(mcp_servers)) or 'none'
                raise UnknownMCPServerReferenceError(
                    f'{base}.mcp[{index}].server: unknown mcp server {mcp.server!r}; '
                    f'available: {available}',
                )


def build_sections(registry: ToolRegistry) -> Iterator[SectionSpec[BaseModel]]:
    # NOTE: the validator needs the runtime registry to check tool overlays
    yield replace(AGENTS, validate=lambda config: _validate_agents(config, registry))
    yield DEFAULTS


def resolve_model(explicit: str | None, default: str | None) -> str:
    if (model := explicit or default) is not None:
        return model

    raise MissingModelError('no model configured; set the model or defaults.model')


AGENTS = SectionSpec(
    'agents',
    AgentConfigList,
    required=True,
    describe=_describe_agents,
)

DEFAULTS = SectionSpec('defaults', DefaultsConfig)
