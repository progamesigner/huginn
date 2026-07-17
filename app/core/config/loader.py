import json
import os
import re
from pathlib import Path
from typing import cast

import yaml

from app.core.tools import ToolRegistry, UnknownToolError

from .exceptions import (
    AmbiguousInstructionsError,
    InvalidValueReferenceError,
    MissingConfigFileError,
    MissingEnvironmentVariableError,
    UnknownAgentReferenceError,
    UnknownMcpServerReferenceError,
    UnknownToolReferenceError,
)
from .schema import HuginnConfig

_ENV_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


def interpolate(text: str, path: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise MissingEnvironmentVariableError(
                f'{path}: environment variable {name!r} is not set'
            )

        return os.environ[name]

    return _ENV_PATTERN.sub(replace, text)


def resolve_value_from(reference: object, path: str) -> object:
    match reference:
        case {'env': str(name), **rest} if not rest:
            if name not in os.environ:
                raise MissingEnvironmentVariableError(
                    f'{path}: environment variable {name!r} is not set'
                )

            return os.environ[name]
        case {'file': str(file_path), **rest} if not rest:
            try:
                return Path(file_path).read_text().strip()
            except OSError as error:
                raise MissingConfigFileError(
                    f'{path}: file {file_path!r} does not exist'
                ) from error
        case _:
            raise InvalidValueReferenceError(
                f'{path}: value_from must be exactly one of {{env: NAME}} or {{file: PATH}}'
            )


def resolve_references(node: object, path: str = '') -> object:
    if isinstance(node, dict):
        mapping = cast(dict[str, object], node)
        if mapping.keys() == {'value_from'}:
            return resolve_value_from(mapping['value_from'], path)

        return {
            key: resolve_references(value, f'{path}.{key}' if path else f'{key}')
            for key, value in mapping.items()
        }

    if isinstance(node, list):
        return [
            resolve_references(item, f'{path}[{index}]')
            for index, item in enumerate(node)
        ]

    if isinstance(node, str):
        return interpolate(node, path)

    return node


def read_yaml(path: Path) -> object:
    try:
        text = path.read_text()
    except OSError as error:
        raise MissingConfigFileError(
            f'config file {str(path)!r} does not exist'
        ) from error

    return yaml.safe_load(text) or {}


def validate_references(config: HuginnConfig, registry: ToolRegistry) -> None:
    for agent_name, agent in config.agents.items():
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
            if mcp.server not in config.mcp_servers:
                available = ', '.join(sorted(config.mcp_servers)) or 'none'
                raise UnknownMcpServerReferenceError(
                    f'{base}.mcp[{index}].server: unknown mcp server {mcp.server!r}; '
                    f'available: {available}'
                )

    if (telegram := config.channels.telegram) is not None:
        for chat_id, agent_name in telegram.bindings.items():
            if agent_name not in config.agents:
                raise UnknownAgentReferenceError(
                    f'channels.telegram.bindings.{chat_id}: unknown agent {agent_name!r}'
                )

        if (
            telegram.router is not None
            and telegram.router.fallback not in config.agents
        ):
            raise UnknownAgentReferenceError(
                f'channels.telegram.router.fallback: unknown agent '
                f'{telegram.router.fallback!r}'
            )


def load_config(path: str | Path, registry: ToolRegistry) -> HuginnConfig:
    raw = read_yaml(Path(path))
    resolved = resolve_references(raw)
    # NOTE: JSON-mode validation so strict mode accepts YAML's plain lists for
    # tuple fields; model_validate()'s python-mode would reject list-for-tuple.
    config = HuginnConfig.model_validate_json(json.dumps(resolved))
    validate_references(config, registry)
    return config
