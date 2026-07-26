from typing import Annotated, Literal

from pydantic import Field, HttpUrl, RootModel, SecretStr

from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue
from app.features.agents.config import AGENTS

from .exceptions import UnknownMCPServerReferenceError

type Transport = Literal['http']


class MCPServerConfig(ConfigValue):
    transport: Annotated[Transport, Field()]
    url: Annotated[HttpUrl, Field()]
    auth: Annotated[SecretStr | None, Field()] = None


class MCPServerConfigList(RootModel[dict[str, MCPServerConfig]]):
    pass


def _describe_mcp_servers(config: HuginnConfig) -> list[str]:
    servers = config.section(MCP_SERVERS).root
    return [f'mcp_servers: {", ".join(sorted(servers)) or "none"}']


def _validate_agent_mcp_refs(config: HuginnConfig) -> None:
    servers = config.section(MCP_SERVERS).root

    for agent_name, agent in config.section(AGENTS).root.items():
        for index, ref in enumerate(agent.mcp):
            if ref.server not in servers:
                available = ', '.join(sorted(servers)) or 'none'
                raise UnknownMCPServerReferenceError(
                    f'agents.{agent_name}.mcp[{index}].server:'
                    f' unknown mcp server {ref.server!r}; available: {available}',
                )


MCP_SERVERS = SectionSpec(
    'mcp_servers',
    MCPServerConfigList,
    describe=_describe_mcp_servers,
    validate=_validate_agent_mcp_refs,
)
