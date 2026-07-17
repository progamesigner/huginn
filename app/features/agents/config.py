from typing import Annotated, Literal, cast

from pydantic import Field, HttpUrl, JsonValue, RootModel, SecretStr

from app.core.config import ConfigValue, HuginnConfig, SectionSpec

from .exceptions import (
    AmbiguousInstructionsError,
    UnknownAgentReferenceError,
    UnknownMcpServerReferenceError,
    UnknownToolReferenceError,
)
from .tools import ToolRegistry, UnknownToolError

type Transport = Literal['http']


class DefaultsConfig(ConfigValue):
    model: Annotated[str | None, Field()] = None


class McpServerConfig(ConfigValue):
    transport: Annotated[Transport, Field()]
    url: Annotated[HttpUrl, Field()]
    auth: Annotated[SecretStr | None, Field()] = None


class ToolRef(ConfigValue):
    name: Annotated[str, Field()]
    params: Annotated[dict[str, JsonValue], Field()] = cast(dict[str, JsonValue], {})


class McpRef(ConfigValue):
    server: Annotated[str, Field()]
    allow: Annotated[tuple[str, ...], Field()] = ()
    inject: Annotated[dict[str, str], Field()] = cast(dict[str, str], {})


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
    bindings: Annotated[dict[str, str], Field()] = cast(dict[str, str], {})
    router: Annotated[ChannelRouterConfig | None, Field()] = None


class ChannelsConfig(ConfigValue):
    telegram: Annotated[ChannelConfig | None, Field()] = None


class AgentsConfig(RootModel[dict[str, AgentConfig]]):
    pass


class McpServersConfig(RootModel[dict[str, McpServerConfig]]):
    pass


def describe_agent(name: str, agent: AgentConfig) -> str:
    model = agent.model or 'default'
    tools = ', '.join(tool.name for tool in agent.tools) or 'none'
    servers = ', '.join(ref.server for ref in agent.mcp) or 'none'
    return f'  {name}: model={model} tools={tools} mcp={servers}'


def describe_agents(config: HuginnConfig) -> list[str]:
    # NOTE: instructions are secrets by policy; the summary must never echo them
    agents = config.section('agents', AgentsConfig).root
    lines = [f'agents ({len(agents)}):']
    lines.extend(describe_agent(name, agent) for name, agent in agents.items())
    return lines


def describe_mcp_servers(config: HuginnConfig) -> list[str]:
    servers = config.section('mcp_servers', McpServersConfig).root
    return [f'mcp_servers: {", ".join(sorted(servers)) or "none"}']


def describe_channels(config: HuginnConfig) -> list[str]:
    channels = config.section('channels', ChannelsConfig)
    if channels.telegram is None:
        return []

    router = channels.telegram.router.fallback if channels.telegram.router else 'none'
    bindings = len(channels.telegram.bindings)
    return ['channels:', f'  telegram: bindings={bindings} router-fallback={router}']


def validate_agents(config: HuginnConfig, registry: ToolRegistry) -> None:
    mcp_servers = config.section('mcp_servers', McpServersConfig).root

    for agent_name, agent in config.section('agents', AgentsConfig).root.items():
        base = f'agents.{agent_name}'

        if (agent.instructions is None) == (agent.instructions_file is None):
            raise AmbiguousInstructionsError(
                f'{base}: exactly one of instructions/instructions_file is required'
            )

        for index, tool in enumerate(agent.tools):
            try:
                registry.get(tool.name)
            except UnknownToolError as error:
                raise UnknownToolReferenceError(
                    f'{base}.tools[{index}].name: {error}'
                ) from error

        for index, mcp in enumerate(agent.mcp):
            if mcp.server not in mcp_servers:
                available = ', '.join(sorted(mcp_servers)) or 'none'
                raise UnknownMcpServerReferenceError(
                    f'{base}.mcp[{index}].server: unknown mcp server {mcp.server!r}; '
                    f'available: {available}'
                )


def validate_channels(config: HuginnConfig) -> None:
    agents = config.section('agents', AgentsConfig).root

    if (telegram := config.section('channels', ChannelsConfig).telegram) is not None:
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


def build_sections(registry: ToolRegistry) -> tuple[SectionSpec, ...]:
    # TODO: channels is contributed here until a channel feature exists (M2)
    return (
        SectionSpec(
            'agents',
            AgentsConfig,
            required=True,
            describe=describe_agents,
            validate=lambda config: validate_agents(config, registry),
        ),
        SectionSpec('defaults', DefaultsConfig),
        SectionSpec('mcp_servers', McpServersConfig, describe=describe_mcp_servers),
        SectionSpec(
            'channels',
            ChannelsConfig,
            describe=describe_channels,
            validate=validate_channels,
        ),
    )
