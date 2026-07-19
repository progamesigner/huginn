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
    UnknownAgentReferenceError,
    UnknownMcpServerReferenceError,
    UnknownToolReferenceError,
)


class DefaultsConfig(ConfigValue):
    model: Annotated[str | None, Field()] = None


class ToolRef(ConfigValue):
    name: Annotated[str, Field()]
    params: Annotated[dict[str, JsonValue], Field()] = Field(default_factory=dict)
    inject: Annotated[dict[str, JsonValue], Field()] = Field(default_factory=dict)


class McpRef(ConfigValue):
    server: Annotated[str, Field()]
    allow: Annotated[tuple[str, ...], Field()] = ()
    inject: Annotated[dict[str, str], Field()] = Field(default_factory=dict)


class AgentConfig(ConfigValue):
    description: Annotated[str, Field()]
    model: Annotated[str | None, Field()] = None
    instructions: Annotated[str | None, Field()] = None
    instructions_file: Annotated[str | None, Field()] = None
    tools: Annotated[tuple[ToolRef, ...], Field()] = ()
    mcp: Annotated[tuple[McpRef, ...], Field()] = ()


class ChannelRouterConfig(ConfigValue):
    fallback: Annotated[str, Field()]
    model: Annotated[str | None, Field()] = None


class ChannelConfig(ConfigValue):
    bindings: Annotated[dict[str, str], Field()] = Field(default_factory=dict)
    router: Annotated[ChannelRouterConfig | None, Field()] = None


class ChannelsConfig(ConfigValue):
    telegram: Annotated[ChannelConfig | None, Field()] = None


class AgentsConfig(RootModel[dict[str, AgentConfig]]):
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


def _describe_channels(config: HuginnConfig) -> list[str]:
    channels = config.section(CHANNELS)
    if channels.telegram is None:
        return []

    router = channels.telegram.router.fallback if channels.telegram.router else 'none'
    bindings = len(channels.telegram.bindings)
    return ['channels:', f'  telegram: bindings={bindings} router-fallback={router}']


def _validate_agents(config: HuginnConfig, registry: ToolRegistry) -> None:
    mcp_servers = config.section(MCP_SERVERS).root
    tools = config.section(TOOLS).root

    for agent_name, agent in config.section(AGENTS).root.items():
        base = f'agents.{agent_name}'

        if (agent.instructions is None) == (agent.instructions_file is None):
            raise AmbiguousInstructionsError(
                f'{base}: exactly one of instructions/instructions_file is required'
            )

        for index, tool in enumerate(agent.tools):
            if tool.name not in tools:
                available = ', '.join(sorted(tools)) or 'none'
                raise UnknownToolReferenceError(
                    f'{base}.tools[{index}].name: unknown tool {tool.name!r}; '
                    f'available: {available}'
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
                raise UnknownMcpServerReferenceError(
                    f'{base}.mcp[{index}].server: unknown mcp server {mcp.server!r}; '
                    f'available: {available}'
                )


def _validate_channels(config: HuginnConfig) -> None:
    agents = config.section(AGENTS).root

    if (telegram := config.section(CHANNELS).telegram) is not None:
        for chat_id, agent_name in telegram.bindings.items():
            if agent_name not in agents:
                raise UnknownAgentReferenceError(
                    f'channels.telegram.bindings.{chat_id}: unknown agent {agent_name!r}'
                )

        if telegram.router is not None and telegram.router.fallback not in agents:
            raise UnknownAgentReferenceError(
                f'channels.telegram.router.fallback: unknown agent '
                f'{telegram.router.fallback!r}'
            )


def build_sections(registry: ToolRegistry) -> Iterator[SectionSpec[BaseModel]]:
    # NOTE: the validator needs the runtime registry to check tool overlays
    yield replace(AGENTS, validate=lambda config: _validate_agents(config, registry))
    yield DEFAULTS
    yield CHANNELS


AGENTS = SectionSpec(
    'agents',
    AgentsConfig,
    required=True,
    describe=_describe_agents,
)
# TODO: channels is contributed here until a channel feature exists
CHANNELS = SectionSpec(
    'channels',
    ChannelsConfig,
    describe=_describe_channels,
    validate=_validate_channels,
)
DEFAULTS = SectionSpec('defaults', DefaultsConfig)
