from dataclasses import dataclass
from typing import Literal

LEGACY_PREFIX = b'agent:memory:'
BIFROST_SEMANTIC_CACHE_PREFIX = b'BifrostSemanticCachePlugin:'


@dataclass(frozen=True)
class AuditReport:
    total: int
    legacy_sessions: int
    bifrost_cache: int
    other: tuple[str, ...]


def classify_key(key: bytes) -> Literal['legacy_sessions', 'bifrost_cache', 'other']:
    if key.startswith(LEGACY_PREFIX):
        return 'legacy_sessions'
    if key.startswith(BIFROST_SEMANTIC_CACHE_PREFIX):
        return 'bifrost_cache'
    return 'other'


def render_key(key: bytes) -> str:
    return key.decode(errors='backslashreplace')
