import json
import os
import re
from pathlib import Path

import yaml
from pydantic import JsonValue

from .exceptions import (
    InvalidConfigError,
    InvalidValueReferenceError,
    MissingConfigFileError,
    MissingEnvironmentVariableError,
    MissingSectionError,
    UnknownSectionError,
)
from .registry import HuginnConfig, SectionRegistry

_ENV_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


def _interpolate(text: str, path: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise MissingEnvironmentVariableError(
                f'{path}: environment variable {name!r} is not set'
            )

        return os.environ[name]

    return _ENV_PATTERN.sub(replace, text)


def _resolve_value_from(reference: JsonValue, path: str) -> JsonValue:
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


def _resolve_references(node: JsonValue, path: str = '') -> JsonValue:
    if isinstance(node, dict):
        if node.keys() == {'value_from'}:
            return _resolve_value_from(node['value_from'], path)

        return {
            key: _resolve_references(value, f'{path}.{key}' if path else f'{key}')
            for key, value in node.items()
        }

    if isinstance(node, list):
        return [
            _resolve_references(item, f'{path}[{index}]')
            for index, item in enumerate(node)
        ]

    if isinstance(node, str):
        return _interpolate(node, path)

    return node


def _read_yaml(path: Path) -> JsonValue:
    try:
        text = path.read_text()
    except OSError as error:
        raise MissingConfigFileError(
            f'config file {str(path)!r} does not exist'
        ) from error

    return yaml.safe_load(text) or {}


def _parse_sections(
    resolved: dict[str, JsonValue],
    registry: SectionRegistry,
) -> HuginnConfig:
    if unknown := sorted(key for key in resolved if key not in registry):
        known = ', '.join(sorted(spec.key for spec in registry)) or 'none'
        raise UnknownSectionError(
            f'unknown config sections: {", ".join(unknown)}; known: {known}'
        )

    sections = {}
    for spec in registry:
        node = resolved.get(spec.key)
        if node is None:
            if spec.required:
                raise MissingSectionError(f'{spec.key}: section is required')

            node = {}

        # NOTE: Use JSON-mode so strict mode accepts YAML's plain lists as tuples
        sections[spec.key] = spec.resolved.model_validate_json(json.dumps(node))

    return HuginnConfig(sections)


def load_config(path: str | Path, registry: SectionRegistry) -> HuginnConfig:
    raw = _read_yaml(Path(path))
    resolved = _resolve_references(raw)
    if not isinstance(resolved, dict):
        raise InvalidConfigError('config root must be a mapping of sections')

    config = _parse_sections(resolved, registry)

    for spec in registry:
        if spec.validate is not None:
            spec.validate(config)

    return config
