from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .contracts import ExecutionContext

_CURRENT: ContextVar[ExecutionContext | None] = ContextVar(
    'agent_execution',
    default=None,
)


def current_execution() -> ExecutionContext | None:
    return _CURRENT.get()


@contextmanager
def execution(context: ExecutionContext) -> Iterator[None]:
    token = _CURRENT.set(context)
    try:
        yield
    finally:
        _CURRENT.reset(token)
