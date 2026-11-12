from typing import Annotated, Literal, Self

from pydantic import Field, HttpUrl, JsonValue, RootModel, model_validator

from app.core.config.exceptions import ConfigError
from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.config.schema import ConfigValue
from app.features.agents.config import AGENTS, BindingSource
from app.features.mcp.config import MCP_SERVERS
from app.features.mcp.exceptions import UnknownMCPServerReferenceError


class BoundHookConfig(ConfigValue):
    bind: Annotated[dict[str, BindingSource], Field()] = Field(default_factory=dict)
    cache_ttl: Annotated[int | None, Field(gt=0)] = None


class MCPHookConfig(BoundHookConfig):
    kind: Annotated[Literal['mcp'], Field()] = 'mcp'
    server: Annotated[str, Field(min_length=1)]
    tool: Annotated[str, Field(min_length=1)]
    arguments: Annotated[dict[str, JsonValue], Field()] = Field(default_factory=dict)

    @model_validator(mode='after')
    def disjoint_arguments(self) -> Self:
        if overlap := sorted(self.arguments.keys() & self.bind.keys()):
            raise ValueError(f'arguments and bind overlap: {", ".join(overlap)}')
        return self


class HTTPHookConfig(BoundHookConfig):
    kind: Annotated[Literal['http'], Field()]
    url: Annotated[HttpUrl, Field()]
    method: Annotated[Literal['GET', 'POST'], Field()] = 'GET'
    headers: Annotated[dict[str, str], Field()] = Field(default_factory=dict)
    timeout: Annotated[int, Field(gt=0)] = 30


class ShellHookConfig(BoundHookConfig):
    kind: Annotated[Literal['shell'], Field()]
    command: Annotated[tuple[str, ...], Field(min_length=1)]
    inherit_env: Annotated[tuple[str, ...], Field()] = ()
    timeout: Annotated[int, Field(gt=0)] = 30

    @model_validator(mode='after')
    def valid_environment_keys(self) -> Self:
        keys = [key.upper() for key in self.bind]
        if len(set(keys)) != len(keys) or any(
            not key
            or not key.isascii()
            or not all(c.isalnum() or c == '_' for c in key)
            for key in keys
        ):
            raise ValueError(
                'shell bind keys must be distinct environment variable names',
            )
        return self


type HookConfig = Annotated[
    MCPHookConfig | HTTPHookConfig | ShellHookConfig,
    Field(discriminator='kind'),
]


class HookConfigList(RootModel[dict[str, HookConfig]]):
    pass


def _validate_hooks(config: HuginnConfig) -> None:
    hooks = config.section(HOOKS).root
    for name, hook in hooks.items():
        if (
            isinstance(hook, MCPHookConfig)
            and hook.server not in config.section(MCP_SERVERS).root
        ):
            raise UnknownMCPServerReferenceError(
                f'hooks.{name}.server: unknown mcp server {hook.server!r}',
            )
    for name, agent in config.section(AGENTS).root.items():
        for hook in agent.hooks:
            if hook not in hooks:
                raise ConfigError(f'agents.{name}.hooks: unknown hook {hook!r}')


HOOKS = SectionSpec('hooks', HookConfigList, validate=_validate_hooks)
