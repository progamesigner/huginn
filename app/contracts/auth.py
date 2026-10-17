from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

type RouterEngine = Literal['llm', 'decision']


@dataclass(frozen=True)
class ApiRouting:
    fallback: str
    engine: RouterEngine = 'llm'
    history_turns: int | None = None
    model: str | None = None
    provider: str | None = None
    allowed: frozenset[str] = field(default_factory=frozenset)
    min_confidence: float | None = None
    min_margin: float | None = None
    shadow_model: str | None = None
    shadow_provider: str | None = None


@dataclass(frozen=True)
class ApiClient:
    key: str
    agents: frozenset[str]
    agent: str | None = None
    routing: ApiRouting | None = None
    denied_tools: frozenset[str] = field(default_factory=frozenset)
    asserted_user: str | None = None


class AuthContract(ABC):
    @abstractmethod
    def authenticate(self, credential: str) -> ApiClient | None: ...
