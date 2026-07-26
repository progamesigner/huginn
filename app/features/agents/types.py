from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SelectionPolicy:
    fallback: str
    model: str | None = None


@dataclass(frozen=True)
class RoutingPolicy:
    bindings: Mapping[str, str] = field(default_factory=dict)
    agent: str | None = None
    selection: SelectionPolicy | None = None
