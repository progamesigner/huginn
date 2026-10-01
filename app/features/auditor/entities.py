from typing import Annotated, Literal

from pydantic import Field

from app.core.schema import AppValue

LEGACY_PREFIX = b'agent:memory:'
BIFROST_SEMANTIC_CACHE_PREFIX = b'BifrostSemanticCachePlugin:'


class AuditReport(AppValue):
    total: Annotated[int, Field()]
    legacy_sessions: Annotated[int, Field()]
    bifrost_cache: Annotated[int, Field()]
    other: Annotated[tuple[str, ...], Field()]


def classify_key(key: bytes) -> Literal['legacy_sessions', 'bifrost_cache', 'other']:
    if key.startswith(LEGACY_PREFIX):
        return 'legacy_sessions'
    if key.startswith(BIFROST_SEMANTIC_CACHE_PREFIX):
        return 'bifrost_cache'
    return 'other'


def render_key(key: bytes) -> str:
    return key.decode(errors='backslashreplace')
