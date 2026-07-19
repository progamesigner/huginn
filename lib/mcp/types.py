from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import JsonValue


@dataclass(frozen=True)
class McpServer:
    name: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class McpToolInfo:
    name: str
    description: str
    input_schema: JsonValue
