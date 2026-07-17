from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, JsonValue, SecretStr

type Transport = Literal['http']


class ConfigValue(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra='forbid')


class DefaultsConfig(ConfigValue):
    model: Annotated[str | None, Field()] = None


class McpServerConfig(ConfigValue):
    transport: Annotated[Transport, Field()]
    url: Annotated[HttpUrl, Field()]
    auth: Annotated[SecretStr | None, Field()] = None


class ToolRef(ConfigValue):
    name: Annotated[str, Field()]
    params: Annotated[dict[str, JsonValue], Field()] = {}


class McpRef(ConfigValue):
    server: Annotated[str, Field()]
    allow: Annotated[tuple[str, ...], Field()] = ()
    inject: Annotated[dict[str, str], Field()] = {}


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
    bindings: Annotated[dict[str, str], Field()] = {}
    router: Annotated[ChannelRouterConfig | None, Field()] = None


class ChannelsConfig(ConfigValue):
    telegram: Annotated[ChannelConfig | None, Field()] = None


class HuginnConfig(ConfigValue):
    agents: Annotated[dict[str, AgentConfig], Field()]
    defaults: Annotated[DefaultsConfig, Field()] = DefaultsConfig()
    mcp_servers: Annotated[dict[str, McpServerConfig], Field()] = {}
    channels: Annotated[ChannelsConfig, Field()] = ChannelsConfig()
