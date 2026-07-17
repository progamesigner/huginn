import json
from pathlib import Path
from typing import Annotated

from pydantic import ValidationError

from app.core.settings import HuginnSettings
from app.core.tools import ToolRegistry
from lib.command import Argument, Exit, Typer, echo
from lib.container import ContainerContract

from .exceptions import ConfigError
from .loader import load_config
from .schema import AgentConfig, ChannelConfig, HuginnConfig


def describe_agent(name: str, agent: AgentConfig) -> str:
    model = agent.model or 'default'
    tools = ', '.join(tool.name for tool in agent.tools) or 'none'
    servers = ', '.join(ref.server for ref in agent.mcp) or 'none'
    return f'  {name}: model={model} tools={tools} mcp={servers}'


def describe_channel(name: str, channel: ChannelConfig) -> str:
    router = channel.router.fallback if channel.router else 'none'
    return f'  {name}: bindings={len(channel.bindings)} router-fallback={router}'


def render_summary(config: HuginnConfig) -> list[str]:
    # NOTE: instructions are secrets by policy
    lines = [f'agents ({len(config.agents)}):']
    lines.extend(describe_agent(name, agent) for name, agent in config.agents.items())

    servers = ', '.join(sorted(config.mcp_servers)) or 'none'
    lines.append(f'mcp_servers: {servers}')

    if config.channels.telegram is not None:
        lines.append('channels:')
        lines.append(describe_channel('telegram', config.channels.telegram))

    return lines


def build_commands(container: ContainerContract) -> Typer:
    commands = Typer(no_args_is_help=True, help='Inspect and validate huginn.yaml.')

    @commands.command()
    def validate(
        path: Annotated[Path | None, Argument(help='Path to huginn.yaml.')] = None,
    ) -> None:
        target = path or container.resolve(HuginnSettings).config_file
        try:
            config = load_config(target, container.resolve(ToolRegistry))
        except (ConfigError, ValidationError) as error:
            echo(str(error), err=True)
            raise Exit(1) from error

        for line in render_summary(config):
            echo(line)

    @commands.command()
    def schema() -> None:
        echo(json.dumps(HuginnConfig.model_json_schema(), indent=2))

    return commands
