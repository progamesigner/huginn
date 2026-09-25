from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ApiRouting:
    fallback: str
    history_turns: int | None = None
    model: str | None = None
    provider: str | None = None
    allowed: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ApiClient:
    key: str
    agents: frozenset[str]
    agent: str | None = None
    routing: ApiRouting | None = None
    denied_tools: frozenset[str] = field(default_factory=frozenset)


class AuthContract(ABC):
    @abstractmethod
    def authenticate(self, credential: str) -> ApiClient | None: ...
