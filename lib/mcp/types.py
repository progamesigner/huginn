from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import JsonValue


@dataclass(frozen=True)
class MCPServer:
    name: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPToolInfo:
    name: str
    description: str
    input_schema: JsonValue
