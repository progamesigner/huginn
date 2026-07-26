from collections.abc import Iterator, MutableMapping

import pytest
from pydantic import JsonValue
from starlette.types import ASGIApp, Receive, Scope, Send
from structlog.contextvars import clear_contextvars, get_contextvars

from app.core.middleware import RequestContextMiddleware


@pytest.fixture(autouse=True)
def isolate_contextvars() -> Iterator[None]:
    clear_contextvars()
    yield
    clear_contextvars()


async def receive() -> MutableMapping[str, JsonValue]:
    return {'type': 'http.request'}


async def send(message: MutableMapping[str, JsonValue]) -> None:
    return None


def capturing_app(seen: dict[str, JsonValue]) -> ASGIApp:
    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        seen.update(get_contextvars())

    return inner


async def test_binds_request_id_for_http_requests() -> None:
    seen = {}
    middleware = RequestContextMiddleware(capturing_app(seen))

    await middleware({'type': 'http', 'path': '/anything'}, receive, send)

    assert seen['request_id']


async def test_binds_fresh_request_id_per_request() -> None:
    first = {}
    second = {}

    await RequestContextMiddleware(capturing_app(first))(
        {'type': 'http', 'path': '/anything'},
        receive,
        send,
    )
    await RequestContextMiddleware(capturing_app(second))(
        {'type': 'http', 'path': '/anything'},
        receive,
        send,
    )

    assert first['request_id'] != second['request_id']


@pytest.mark.parametrize('path', ['/healthz', '/readyz'])
async def test_skips_excluded_paths(path: str) -> None:
    seen = {}
    middleware = RequestContextMiddleware(
        capturing_app(seen),
        excludes=frozenset({'/healthz', '/readyz'}),
    )

    await middleware({'type': 'http', 'path': path}, receive, send)

    assert 'request_id' not in seen


async def test_binds_request_id_when_no_excludes_configured() -> None:
    seen = {}
    middleware = RequestContextMiddleware(capturing_app(seen))

    await middleware({'type': 'http', 'path': '/healthz'}, receive, send)

    assert seen['request_id']


async def test_passes_lifespan_scope_through() -> None:
    seen = {}
    middleware = RequestContextMiddleware(capturing_app(seen))

    await middleware({'type': 'lifespan'}, receive, send)

    assert 'request_id' not in seen
