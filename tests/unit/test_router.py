from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

import fastapi
import fastapi.params
import fastapi.responses
import httpx
from fastapi import APIRouter, FastAPI
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from lib.router import (
    Depends,
    PlainTextResponse,
    Request,
    Response,
    Router,
    RouterManager,
)


def probe_router(path: str = '/ping') -> APIRouter:
    router = Router()

    @router.get(path)
    async def ping() -> dict[str, str]:
        return {'status': 'ok'}

    return router


async def fetch(app: ASGIApp, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://huginn',
    ) as client:
        return await client.get(path)


def test_reexports_are_fastapi_primitives() -> None:
    assert Router is APIRouter
    assert Depends is fastapi.params.Depends
    assert Depends is not fastapi.Depends
    assert Request is fastapi.Request
    assert Response is fastapi.Response


def test_param_reexports_are_fastapi_params_classes() -> None:
    import lib.router as router

    names = (
        'Body',
        'Cookie',
        'Depends',
        'File',
        'Form',
        'Header',
        'Param',
        'ParamTypes',
        'Path',
        'Query',
        'Security',
    )
    for name in names:
        assert getattr(router, name) is getattr(fastapi.params, name)


def test_response_reexports_are_fastapi_responses() -> None:
    import lib.router as router

    names = (
        'EventSourceResponse',
        'FileResponse',
        'HTMLResponse',
        'JSONResponse',
        'PlainTextResponse',
        'RedirectResponse',
        'Response',
        'StreamingResponse',
    )
    for name in names:
        assert getattr(router, name) is getattr(fastapi.responses, name)


async def test_include_registers_routes() -> None:
    manager = RouterManager()
    manager.include(probe_router())

    response = await fetch(manager.app, '/ping')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


async def test_include_after_build_registers_routes() -> None:
    manager = RouterManager()
    manager.include(probe_router('/ping'))

    assert (await fetch(manager.app, '/ping')).status_code == 200

    manager.include(probe_router('/pong'))

    assert (await fetch(manager.app, '/pong')).status_code == 200


async def test_docs_enabled_exposes_openapi() -> None:
    manager = RouterManager(docs=True)
    manager.include(probe_router())

    assert (await fetch(manager.app, '/docs')).status_code == 200
    assert (await fetch(manager.app, '/openapi.json')).status_code == 200


async def test_docs_disabled_hides_openapi() -> None:
    manager = RouterManager(docs=False)
    manager.include(probe_router())

    assert (await fetch(manager.app, '/docs')).status_code == 404
    assert (await fetch(manager.app, '/openapi.json')).status_code == 404
    assert (await fetch(manager.app, '/ping')).status_code == 200


def test_debug_flag_reaches_app() -> None:
    assert cast(FastAPI, RouterManager(debug=True).app).debug is True
    assert cast(FastAPI, RouterManager(debug=False).app).debug is False


async def test_middlewares_wrap_the_app() -> None:
    @dataclass(frozen=True)
    class MarkMiddleware:
        app: ASGIApp

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            async def marked(message: Message) -> None:
                if message['type'] == 'http.response.start':
                    message['headers'].append((b'x-mark', b'1'))
                await send(message)

            await self.app(scope, receive, marked)

    manager = RouterManager(middlewares=[Middleware(MarkMiddleware)])
    manager.include(probe_router())

    response = await fetch(manager.app, '/ping')

    assert response.headers['x-mark'] == '1'


async def test_exception_handlers_are_wired_to_the_app() -> None:
    async def handle_value_error(_: Request, error: ValueError) -> Response:
        return PlainTextResponse(f'handled: {error}', status_code=400)

    router = probe_router()

    @router.get('/explode')
    async def explode() -> None:
        raise ValueError('boom')

    manager = RouterManager(exception_handlers={ValueError: handle_value_error})
    manager.include(router)

    response = await fetch(manager.app, '/explode')

    assert response.status_code == 400
    assert response.text == 'handled: boom'


async def test_lifespan_is_entered_and_exited() -> None:
    events = []

    @asynccontextmanager
    async def lifespan(_: object) -> AsyncIterator[None]:
        events.append('enter')
        yield
        events.append('exit')

    manager = RouterManager(lifespan=lifespan)
    app = cast(Starlette, manager.app)

    async with app.router.lifespan_context(app):
        events.append('serving')

    assert events == ['enter', 'serving', 'exit']
