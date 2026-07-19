from typing import Annotated, Literal

from pydantic import Field, HttpUrl, RootModel, SecretStr

from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue

type Transport = Literal['http']


class McpServerConfig(ConfigValue):
    transport: Annotated[Transport, Field()]
    url: Annotated[HttpUrl, Field()]
    auth: Annotated[SecretStr | None, Field()] = None


class McpServersConfig(RootModel[dict[str, McpServerConfig]]):
    pass


def _describe_mcp_servers(config: HuginnConfig) -> list[str]:
    servers = config.section(MCP_SERVERS).root
    return [f'mcp_servers: {", ".join(sorted(servers)) or "none"}']


MCP_SERVERS = SectionSpec(
    'mcp_servers',
    McpServersConfig,
    describe=_describe_mcp_servers,
)
